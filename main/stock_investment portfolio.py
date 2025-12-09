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
        df = pd.read_excel(f'{sym}.xlsx')
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