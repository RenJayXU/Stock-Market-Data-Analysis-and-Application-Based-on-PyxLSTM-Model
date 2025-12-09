import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import warnings

# 設定繪圖風格
plt.style.use('ggplot')
warnings.simplefilter("ignore")

def main():
    # 1. 定義股票代號
    symbols = ['1301', '2330', '2882', '1734', '3008']
    
    # 設定無風險利率 (可查詢目前美債或台債殖利率)
    RISK_FREE_RATE = 0.02 

    # --- A. 讀取與處理預測資料 (計算預期報酬 mu) ---
    print(">>> 讀取預測資料 (Calculating Expected Returns)...")
    
    # 建立一個空的 DataFrame 來存放所有股票的預測價格
    # 使用 merge 確保日期對齊
    predicted_prices = None

    for sym in symbols:
        # 假設預測檔名為 future_prediction_2330.csv (請確保與 predict.py 輸出一致)
        # 如果您的檔名不同，請在此修改
        pred_filename = f"processed_data/test_{sym}.csv" # 這裡示範讀取測試集結果，或您自定義的預測檔
        # 注意：實際應用中，這裡應該讀取 stock_predict.py 產出的 "未來 N 天預測值"
        # 為了演示，我們假設您有一個格式為 [Date, Predicted_Close] 的 CSV
        
        # 這裡我們模擬一個讀取邏輯，請根據您實際的檔案路徑修改
        # 假設檔案在 results 資料夾或是根目錄
        possible_paths = [
            f"results/future_prediction_{sym}.csv", 
            f"future_prediction{sym}.csv",
            f"future_prediction{sym}.xlsx - Sheet1.csv" # 根據您上傳的檔名
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"  - {sym} 讀取成功: {path}")
                break
        
        if df is None:
            print(f"  [錯誤] 找不到 {sym} 的預測檔案，跳過此股票。")
            continue

        # 確保有 Date 欄位
        if 'Date' not in df.columns:
            # 如果沒有 Date，嘗試產生一個 (假設是最近的交易日)
            # 這邊為了安全，若無 Date 則報錯，或者您需要手動加上 Date
             print(f"  [警告] {sym} 的資料中沒有 'Date' 欄位，無法進行日期對齊。")
             continue

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 重新命名 Close 欄位
        temp_df = df[['Date', 'Predicted_Close']].rename(columns={'Predicted_Close': sym})
        
        if predicted_prices is None:
            predicted_prices = temp_df
        else:
            # 關鍵修正：使用 merge 進行 Inner Join，確保日期一致
            predicted_prices = pd.merge(predicted_prices, temp_df, on='Date', how='inner')

    if predicted_prices is None or predicted_prices.empty:
        print("無法建立預測價格表，請檢查檔案路徑。")
        return

    print(f"  - 資料對齊後，共計 {len(predicted_prices)} 個交易日的預測數據。")

    # 計算預測的日報酬率
    # Set index to Date for pct_change
    predicted_prices.set_index('Date', inplace=True)
    predicted_returns = predicted_prices.pct_change().dropna()

    # 計算年化預期報酬 (mu)
    # 注意：這裡假設 xLSTM 的短期預測趨勢能代表整年的平均水準
    mu = predicted_returns.mean() * 252

    # --- B. 讀取歷史資料 (計算風險矩陣 Sigma) ---
    print("\n>>> 讀取歷史資料 (Calculating Risk Matrix)...")
    historical_prices = pd.DataFrame()

    for sym in predicted_prices.columns: # 只處理有預測資料的股票
        try:
            # 讀取原始資料 (假設在 data 資料夾或根目錄)
            path = f"data/{sym}.csv"
            if not os.path.exists(path):
                path = f"{sym}.csv"
            
            hist_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            hist_df = hist_df.sort_index()
            historical_prices[sym] = hist_df['Close']
        except Exception as e:
            print(f"  [錯誤] 讀取 {sym} 歷史資料失敗: {e}")
            return

    # 補齊缺失值
    historical_prices = historical_prices.ffill().dropna()
    
    # 計算歷史日報酬
    historical_returns = historical_prices.pct_change().dropna()

    # 關鍵修正：只使用最近 1 年 (約 252 天) 的資料來計算 Covariance
    # 這樣更能反映當下的市場波動狀況
    lookback_window = 252
    if len(historical_returns) > lookback_window:
        print(f"  - 使用最近 {lookback_window} 天的歷史資料計算風險...")
        recent_returns = historical_returns.iloc[-lookback_window:]
    else:
        print(f"  - 使用全部 {len(historical_returns)} 天的歷史資料計算風險...")
        recent_returns = historical_returns

    # 年化共變異數矩陣 (Sigma)
    Sigma = recent_returns.cov() * 252

    # --- C. 投資組合優化函數 ---
    def neg_sharpe(w, mu, Sigma, rf):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        return -(port_ret - rf) / port_vol if port_vol > 0 else 0

    def get_portfolio_metrics(weights, mu, Sigma, rf):
        ret = np.dot(weights, mu)
        vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        sr = (ret - rf) / vol if vol > 0 else 0
        return ret, vol, sr

    # 優化設定
    n_assets = len(mu)
    args = (mu, Sigma, RISK_FREE_RATE)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = [1./n_assets for _ in range(n_assets)]

    # 執行優化 (最大 Sharpe)
    print("\n>>> 執行優化 (Maximizing Sharpe Ratio)...")
    result = minimize(neg_sharpe, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print("優化失敗！請檢查數據。")
        return

    best_weights = result.x
    final_ret, final_vol, final_sr = get_portfolio_metrics(best_weights, mu, Sigma, RISK_FREE_RATE)

    # --- D. 輸出結果 ---
    print("\n" + "="*40)
    print("      最佳投資組合建議 (xLSTM + Markowitz)")
    print("="*40)
    
    results_df = pd.DataFrame({
        'Stock': mu.index,
        'Weight': best_weights,
        'Exp Return (Pred)': mu.values,
        'Risk (Hist)': [np.sqrt(Sigma.loc[s, s]) for s in mu.index]
    })
    
    # 格式化輸出
    print(results_df.round(4))
    
    print("-" * 40)
    print(f"預期年化報酬: {final_ret:.2%}")
    print(f"預期年化波動: {final_vol:.2%}")
    print(f"Sharpe Ratio: {final_sr:.2f}")
    print("-" * 40)

    # --- E. 視覺化：效率前緣 (Efficient Frontier) ---
    print("\n繪製效率前緣圖...")
    
    # 產生隨機組合用於繪製散佈圖
    num_ports = 5000
    all_weights = np.zeros((num_ports, n_assets))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for i in range(num_ports):
        # 隨機權重
        weights = np.array(np.random.random(n_assets))
        weights /= np.sum(weights)
        all_weights[i,:] = weights
        
        # 計算指標
        ret_arr[i] = np.dot(weights, mu)
        vol_arr[i] = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        sharpe_arr[i] = (ret_arr[i] - RISK_FREE_RATE) / vol_arr[i]

    plt.figure(figsize=(10, 6))
    # 繪製隨機點
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', marker='.', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    
    # 標記最佳組合
    plt.scatter(final_vol, final_ret, c='red', s=100, marker='*', label='Optimal Portfolio')
    
    # 標記單一股票
    for sym in mu.index:
        s_vol = np.sqrt(Sigma.loc[sym, sym])
        s_ret = mu[sym]
        plt.scatter(s_vol, s_ret, c='black', s=50, marker='o')
        plt.text(s_vol, s_ret, f" {sym}", fontsize=9)

    plt.title(f'Efficient Frontier (xLSTM Predictions)\nMax Sharpe: {final_sr:.2f}')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/portfolio_optimization.png"
    plt.savefig(save_path)
    print(f"圖表已儲存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()