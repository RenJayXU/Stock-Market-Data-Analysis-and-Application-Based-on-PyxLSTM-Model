import pandas as pd
import numpy as np
import os
import yfinance as yf
import warnings

warnings.simplefilter("ignore")

def main():
    # ==========================================
    # 🔴 參數設定區 (你可以在這裡自由修改)
    # ==========================================
    INITIAL_CAPITAL = 1000000  # 初始資金：100萬
    TOP_N = 5                  # 選出預測前 5 名
    TARGET_DATE = '2025-01-02' # 想要測試的「買入日期」 (格式: YYYY-MM-DD)
    # ==========================================
    
    symbols = [
        # 前 10 大權重股
    '2330', '2317', '2454', '2308', '2382', '3711', '2303', '2891', '2881', '2882',
    
    # 金融控股族群
    '2886', '2884', '2892', '2890', '5880', '2883', '2887', '2885', '5871', '2880', '5876',
    
    # 電子、半導體與 AI 供應鏈 (含 Q3 新增: 2059)
    '2412', '3045', '4904', '3231', '2357', '2059', '2301', '2345', '3008', '2379', 
    '2408', '3034', '2327', '2474', '2395', '6669',
    
    # 航運與交通 (含 Q1 新增: 2615)
    '2603', '2609', '2615', '2207',
    
    # 生技醫療 (Q3 新增: 6919)
    '6919',
    
    # 塑化、鋼鐵、食品與傳產 (已剔除: 1326, 1101, 1760)
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'
    ]

    pred_returns = {}
    true_prices = {}
    
    # 1. 讀取預測分數與真實股價
    for sym in symbols:
        pred_path = f"results/future_prediction_{sym}.csv"
        price_path = f"data/{sym}.csv"
        
        if os.path.exists(pred_path) and os.path.exists(price_path):
            pred_df = pd.read_csv(pred_path, parse_dates=['Date']).set_index('Date')
            price_df = pd.read_csv(price_path, parse_dates=['Date']).set_index('Date')
            
            pred_returns[sym] = pred_df['Predicted_Return']
            true_prices[sym] = price_df['Close']
            
    df_pred = pd.DataFrame(pred_returns).dropna(how='all')
    df_price = pd.DataFrame(true_prices).dropna(how='all')

    if df_pred.empty:
        print("錯誤：找不到預測資料。")
        return

# 對齊日期：我們只確保預測資料不會超出真實股價的範圍
    # 🔴 注意：絕對不要去裁切 df_price，這樣才能保留未來 20 天的真實股價用來結算！
    common_dates = df_pred.index.intersection(df_price.index)
    df_pred = df_pred.loc[common_dates]
    # 移除 df_price 的裁切

    # 2. 設定買入日 (Day 1) 與賣出日 (Day 20)
    target_date = pd.to_datetime(TARGET_DATE)
    
    # 防呆檢查：確認輸入的日期是否存在於資料庫中
    if target_date not in df_pred.index:
        print(f"\n⚠️ 錯誤：找不到 {TARGET_DATE} 的預測資料。")
        print(f"👉 可能是假日無交易，或超出資料範圍。")
        print(f"👉 目前可用的日期範圍為：{df_pred.index[0].date()} 到 {df_pred.index[-1].date()}\n")
        return
        
    buy_date = target_date
    
    # 尋找 20 個交易日之後的日期
    try:
        buy_idx = df_price.index.get_loc(buy_date)
        sell_idx = buy_idx + 20
        if sell_idx >= len(df_price):
            print(f"⚠️ 警告：從 {buy_date.date()} 起算，尚未滿 20 個交易日，將使用最後一天結算。")
            sell_idx = len(df_price) - 1
        sell_date = df_price.index[sell_idx]
    except KeyError:
        print("日期對齊錯誤。")
        return

    # 3. 抓取這段期間的 0050 作為比較基準
    print("正在下載真實 0050 ETF 數據...")
    benchmark_0050 = yf.download('0050.TW', start=buy_date, end=sell_date + pd.Timedelta(days=1), progress=False)
    
    # 4. Day 1 進行選股與資金分配
    predictions_on_buy_date = df_pred.loc[buy_date]
    positive_preds = predictions_on_buy_date[predictions_on_buy_date > 0]
    
    print(f"\n{'='*50}")
    print(f"📅 買入日: {buy_date.date()}")
    print(f"📅 結算日: {sell_date.date()} (持有 {sell_idx - buy_idx} 個交易日)")
    print(f"{'='*50}")
    
    if len(positive_preds) == 0:
        print("當天沒有看漲的股票，保持空手。")
        return
        
    top_n_stocks = positive_preds.nlargest(TOP_N)
    print("🎯 AI 預測買入名單與權重 (平均分配):")
    weight_per_stock = 1.0 / len(top_n_stocks)
    for stock in top_n_stocks.index:
        print(f" - {stock}: {weight_per_stock:.2%} (預測漲幅: {top_n_stocks[stock]:.2%})")

    # 5. 計算 20 天後的真實獲利
    # 股票報酬率 = (賣出日股價 - 買入日股價) / 買入日股價
    stock_returns = (df_price.loc[sell_date, top_n_stocks.index] - df_price.loc[buy_date, top_n_stocks.index]) / df_price.loc[buy_date, top_n_stocks.index]
    
    # 投資組合總報酬 = 個股報酬的加權平均
    portfolio_return = (stock_returns * weight_per_stock).sum()
    ai_final_capital = INITIAL_CAPITAL * (1 + portfolio_return)
    
    # 0050 總報酬
    bench_start_price = benchmark_0050.loc[benchmark_0050.index >= buy_date, 'Close'].iloc[0]
    bench_end_price = benchmark_0050.loc[benchmark_0050.index <= sell_date, 'Close'].iloc[-1]
    
    # yfinance 回傳可能是 Series (如果只有一檔股票)，確保我們拿到的是數值
    if isinstance(bench_start_price, pd.Series): bench_start_price = bench_start_price.item()
    if isinstance(bench_end_price, pd.Series): bench_end_price = bench_end_price.item()
    
    benchmark_return = (bench_end_price - bench_start_price) / bench_start_price
    bench_final_capital = INITIAL_CAPITAL * (1 + benchmark_return)

    # 6. 輸出終極對決結果
    print(f"\n{'='*50}")
    print(f"💰 單次 {sell_idx - buy_idx} 天持有真實對決 (初始資金: {INITIAL_CAPITAL:,.0f} 元)")
    print(f"{'='*50}")
    print(f"{'項目':<15} | {'xLSTM 組合':<15} | {'0050 ETF':<15}")
    print("-" * 50)
    print(f"{'結算總資產':<15} | {ai_final_capital:>12,.0f} 元 | {bench_final_capital:>12,.0f} 元")
    print(f"{'淨獲利':<15} | {(ai_final_capital - INITIAL_CAPITAL):>12,.0f} 元 | {(bench_final_capital - INITIAL_CAPITAL):>12,.0f} 元")
    print(f"{'區間總報酬率':<14} | {portfolio_return:>14.2%} | {benchmark_return:>14.2%}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()