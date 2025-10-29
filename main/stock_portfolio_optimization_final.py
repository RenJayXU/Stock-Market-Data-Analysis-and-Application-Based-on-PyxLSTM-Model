import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.simplefilter("ignore")

def main():
    # 1. 讀取五檔股票預測價格並統一處理
    symbols = ['1301', '2330', '2882', '1734', '3008']
    stock_data = {}

    print("讀取各檔股票資料:")
    for sym in symbols:
        df = pd.read_excel(f'future_prediction{sym}.xlsx')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        stock_data[sym] = df
        print(f"{sym}: {len(df)} 筆資料, 日期範圍 {df['Date'].min()} 至 {df['Date'].max()}")

    # 2. 由於日期不同，我們直接用序列號來對齊（假設都是連續的60個交易日預測）
    prices = pd.DataFrame()
    for sym in symbols:
        prices[sym] = stock_data[sym]['Predicted_Close'].values

    print(f"\n價格資料形狀: {prices.shape}")
    print("價格資料前5筆：")
    print(prices.head())

    # 3. 計算日報酬率
    returns = prices.pct_change().dropna()
    print(f"\n日報酬率資料形狀: {returns.shape}")

    # 4. 年化預期報酬率與風險
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252

    print(f"\n=== 各股票年化指標 ===")
    for sym in symbols:
        vol = np.sqrt(Sigma.loc[sym, sym])
        print(f"{sym}: 預期報酬={mu[sym]:.2%}, 波動率={vol:.2%}")

    # 5. 投資組合優化
    rf = 0.02  # 無風險利率2%

    def neg_sharpe(w, mu, Sigma, rf):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        if port_vol <= 0:
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

    print(f"\n=== 投資組合優化結果 ===")
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
        print(f"預期年化報酬: {port_ret:.2%}")
        print(f"年化波動率: {port_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # 回測累積回報
        port_daily = returns.dot(weights)
        cumulative = (1 + port_daily).cumprod()
        max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()
        
        print(f"\n回測結果:")
        print(f"60天期間總回報: {(cumulative.iloc[-1] - 1):.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        
    else:
        print(f"優化失敗: {res.message}")

    # 6. 多元化投資組合（限制單一股票不超過30%）
    print(f"\n=== 多元化投資組合 (單一股票≤30%) ===")
    bounds_div = [(0, 0.3)] * n
    res_div = minimize(neg_sharpe, x0, args=(mu, Sigma, rf),
                       method='SLSQP', bounds=bounds_div, constraints=constraints)

    if res_div.success:
        weights_div = pd.Series(res_div.x, index=symbols)
        print(f"\n多元化權重 (%):")
        for sym in symbols:
            print(f"{sym}: {weights_div[sym]*100:.2f}%")
        
        port_ret_div = np.dot(weights_div, mu)
        port_vol_div = np.sqrt(np.dot(weights_div, np.dot(Sigma, weights_div)))
        sharpe_div = (port_ret_div - rf) / port_vol_div
        
        print(f"\n多元化組合績效:")
        print(f"預期年化報酬: {port_ret_div:.2%}")
        print(f"年化波動率: {port_vol_div:.2%}")
        print(f"Sharpe Ratio: {sharpe_div:.2f}")
        
        # 多元化組合回測
        port_daily_div = returns.dot(weights_div)
        cumulative_div = (1 + port_daily_div).cumprod()
        max_drawdown_div = ((cumulative_div / cumulative_div.cummax()) - 1).min()
        
        print(f"\n多元化回測結果:")
        print(f"60天期間總回報: {(cumulative_div.iloc[-1] - 1):.2%}")
        print(f"最大回撤: {max_drawdown_div:.2%}")
    else:
        print(f"多元化優化失敗: {res_div.message}")

    # 7. 等權重基準比較
    equal_weights = pd.Series([0.2] * 5, index=symbols)
    port_ret_equal = np.dot(equal_weights, mu)
    port_vol_equal = np.sqrt(np.dot(equal_weights, np.dot(Sigma, equal_weights)))
    sharpe_equal = (port_ret_equal - rf) / port_vol_equal
    
    print(f"\n=== 等權重基準策略 ===")
    print(f"預期年化報酬: {port_ret_equal:.2%}")
    print(f"年化波動率: {port_vol_equal:.2%}")
    print(f"Sharpe Ratio: {sharpe_equal:.2f}")

if __name__ == "__main__":
    main()