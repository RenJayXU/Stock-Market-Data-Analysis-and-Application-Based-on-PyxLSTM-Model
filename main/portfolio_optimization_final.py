import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.simplefilter("ignore")

def main():
    # 1. 定義股票代號
    symbols = ['1301', '2330', '2882', '1734', '3008']
    
    # --- 計算預期報酬 (mu) ---
    print("讀取各檔股票 預測 資料 (用於計算 mu):")
    stock_data = {}
    for sym in symbols:
        try:
            df = pd.read_excel(f'future_prediction{sym}.xlsx')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            stock_data[sym] = df
            print(f"{sym}: {len(df)} 筆資料, 日期範圍 {df['Date'].min()} 至 {df['Date'].max()}")
        except FileNotFoundError:
            print(f"錯誤：找不到預測檔案 future_prediction{sym}.xlsx。請先執行預測腳本。")
            return

    # 2. 由於日期不同，我們直接用序列號來對齊（假設都是連續的60個交易日預測）
    prices = pd.DataFrame()
    for sym in symbols:
        prices[sym] = stock_data[sym]['Predicted_Close'].values

    # 3. 計算 "預測的" 日報酬率
    predicted_returns = prices.pct_change().dropna()

    # 4. 年化 "預期" 報酬率 (mu)
    # 這是我們使用 xLSTM 模型預測的未來趨勢
    mu = predicted_returns.mean() * 252
    
    # --- 計算風險 (Sigma) ---
    print("\n讀取各檔股票 歷史 資料 (用於計算 Sigma):")
    historical_prices = pd.DataFrame()
    
    for sym in symbols:
        # 假設您的原始CSV檔與此 .py 檔在同一目錄下，且檔名為 {sym}.csv
        try:
            hist_df = pd.read_csv(f"{sym}.csv", parse_dates=["Date"], index_col="Date")
            # 確保資料是按日期排序的
            hist_df = hist_df.sort_index()
            historical_prices[sym] = hist_df['Close']
        except FileNotFoundError:
            print(f"錯誤：找不到歷史資料 {sym}.csv。請確保檔案路徑正確。")
            return
    
    # 補齊缺失值 (例如假日，使用前一天的收盤價)
    historical_prices = historical_prices.ffill()
    
    # 計算 "歷史" 日報酬率
    historical_returns = historical_prices.pct_change().dropna()
    
    # 建議：只使用最近 2 年 (約 504 筆) 的歷史資料來計算風險，更能反映近期市場結構
    if len(historical_returns) > 504:
        print(f"\n使用最近 2 年 (504筆) 的歷史報酬率來計算風險...")
        historical_returns = historical_returns.iloc[-504:]
    else:
        print(f"\n使用全部 {len(historical_returns)} 筆歷史報酬率來計算風險...")

    # 5. 年化 "歷史" 風險 (Sigma)
    # 這是基於真實市場波動和相關性計算的風險矩陣
    Sigma = historical_returns.cov() * 252


    print(f"\n=== 各股票年化指標 (mu來自預測, vol來自歷史) ===")
    for sym in symbols:
        try:
            vol = np.sqrt(Sigma.loc[sym, sym])
            print(f"{sym}: 預期報酬={mu[sym]:.2%}, 歷史波動率={vol:.2%}")
        except KeyError:
            print(f"錯誤：無法在歷史資料中找到 {sym} 的波動率。")
            
    # 6. 投資組合優化
    rf = 0.02  # 無風險利率2%

    def neg_sharpe(w, mu, Sigma, rf):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        if port_vol <= 1e-6: # 避免除以零
            return np.inf
        return -(port_ret - rf) / port_vol

    # 約束和邊界
    n = len(symbols)
    bounds = [(0, 1)] * n  # 不做空，上限100%
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 求解最優權重
    x0 = np.repeat(1/n, n)
    res = minimize(neg_sharpe, x0, args=(mu, Sigma, rf),
                   method='SLSQP', bounds=bounds, constraints=constraints)

    print(f"\n=== 投資組合優化結果 (最大化 Sharpe Ratio) ===")
    if res.success:
        weights = pd.Series(res.x, index=symbols)
        print(f"\n推薦權重 (%):")
        for sym in symbols:
            print(f"{sym}: {weights[sym]*100:.2f}%")
        
        # 投資組合績效
        port_ret = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        sharpe = (port_ret - rf) / port_vol
        
        print(f"\n投資組合績效:")
        print(f"預期年化報酬 (來自預測): {port_ret:.2%}")
        print(f"年化波動率 (來自歷史): {port_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
    else:
        print(f"優化失敗: {res.message}")

    # 7. 多元化投資組合（限制單一股票不超過30%）
    print(f"\n=== 多元化投資組合 (單一股票≤30%) ===")
    bounds_div = [(0, 0.3)] * n
    res_div = minimize(neg_sharpe, x0, args=(mu, Sigma, rf),
                       method='SLSQP', bounds=bounds_div, constraints=constraints)

    if res_div.success:
        weights_div = pd.Series(res_div.x, index=symbols)
        print(f"\n多元化權重 (%):")
        for sym in symbols:
            # 對極小的權重 (例如 1e-17) 視為 0
            weight_val = weights_div[sym] if weights_div[sym] > 1e-6 else 0
            print(f"{sym}: {weight_val*100:.2f}%")
        
        port_ret_div = np.dot(weights_div, mu)
        port_vol_div = np.sqrt(np.dot(weights_div, np.dot(Sigma, weights_div)))
        sharpe_div = (port_ret_div - rf) / port_vol_div
        
        print(f"\n多元化組合績效:")
        print(f"預期年化報酬 (來自預測): {port_ret_div:.2%}")
        print(f"年化波動率 (來自歷史): {port_vol_div:.2%}")
        print(f"Sharpe Ratio: {sharpe_div:.2f}")

    else:
        print(f"多元化優化失敗: {res_div.message}")

    # 8. 等權重基準比較 (使用歷史風險)
    equal_weights = pd.Series([1/n] * n, index=symbols)
    port_ret_equal = np.dot(equal_weights, mu)
    port_vol_equal = np.sqrt(np.dot(equal_weights, np.dot(Sigma, equal_weights)))
    sharpe_equal = (port_ret_equal - rf) / port_vol_equal
    
    print(f"\n=== 等權重基準策略 (mu來自預測, vol來自歷史) ===")
    print(f"預期年化報酬: {port_ret_equal:.2%}")
    print(f"年化波動率: {port_vol_equal:.2%}")
    print(f"Sharpe Ratio: {sharpe_equal:.2f}")

if __name__ == "__main__":
    main()