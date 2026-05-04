import pandas as pd
import numpy as np
import os
import yfinance as yf
import warnings

warnings.simplefilter("ignore")

def main():
    # ==========================================
    # 🔴 參數設定區
    # ==========================================
    INITIAL_CAPITAL = 1000000  # 初始資金：100萬
    TOP_N = 5                  # 選出預測前 5 名
    TARGET_DATE = '2025-07-01' # 第一期的「買入日期」
    NUM_PERIODS = 3            # 滾動次數 (3 次 = 60 個交易日)
    HOLDING_DAYS = 20          # 每個週期的持有交易日
    
    # 🔴 雙重過濾網參數 (Quality Control)
    MIN_R2 = -0.132432554              # R2 必須大於平均
    MIN_ACCURACY = 0.50        # 方向準確率必須大於等於 50%
    
    # 🔴 交易成本參數 (台股標準)
    BROKER_FEE_RATE = 0.001425 # 公定手續費 0.1425%
    FEE_DISCOUNT = 0.5         # 券商手續費折扣 (預設 5 折)
    TRANS_TAX_RATE = 0.003     # 證交稅 0.3% (僅賣出收取)
    
    # 實際費率計算
    ACTUAL_BUY_FEE = BROKER_FEE_RATE * FEE_DISCOUNT
    ACTUAL_SELL_FEE = (BROKER_FEE_RATE * FEE_DISCOUNT) + TRANS_TAX_RATE
    # ==========================================
    
    symbols = [
        '2330', '2317', '2454', '2308', '2382', '3711', '2303', '2891', '2881', '2882',
        '2886', '2884', '2892', '2890', '5880', '2883', '2887', '2885', '5871', '2880', '5876',
        '2412', '3045', '4904', '3231', '2357', '2059', '2301', '2345', '3008', '2379', 
        '2408', '3034', '2327', '2474', '2395', '6669',
        '2603', '2609', '2615', '2207',
        '6919',
        '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'
    ]

    # ==========================================
    # 🔴 1. 讀取模型的「體檢報告」建立品質白名單
    # ==========================================
    metrics_path = "results/all_stocks_metrics_detail.csv"
    qualified_symbols = symbols # 預設全部通過 (防呆)
    
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        # 篩選 R2 與 Accuracy 達標的股票
        qualified_df = df_metrics[(df_metrics['R2'] >= MIN_R2) & (df_metrics['Accuracy'] >= MIN_ACCURACY)]
        qualified_symbols = qualified_df['Stock'].astype(str).tolist()
        print(f"✅ 雙重品質過濾：共有 {len(qualified_symbols)} 檔股票通過 (R2 >= {MIN_R2}, Accuracy >= {MIN_ACCURACY:.0%}) 進入白名單。")
    else:
        print("⚠️ 找不到 metrics 評估檔案 (all_stocks_metrics_detail.csv)，將不進行品質過濾！")

    pred_returns = {}
    true_prices = {}
    
    # 2. 讀取預測分數與真實股價
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

    common_dates = df_pred.index.intersection(df_price.index)
    df_pred = df_pred.loc[common_dates]

    target_date = pd.to_datetime(TARGET_DATE)
    
    if target_date not in df_pred.index:
        print(f"\n⚠️ 錯誤：找不到 {TARGET_DATE} 的預測資料。")
        return
        
    print("正在下載真實 0050 ETF 數據...")
    end_date_approx = target_date + pd.Timedelta(days=(NUM_PERIODS+1)*HOLDING_DAYS*1.5)
    benchmark_0050 = yf.download('0050.TW', start=target_date, end=end_date_approx, progress=False)
    
    bench_start_price = benchmark_0050.loc[benchmark_0050.index >= target_date, 'Close'].iloc[0]
    if isinstance(bench_start_price, pd.Series): bench_start_price = bench_start_price.item()

    # 初始化滾動狀態與 0050 買入成本
    current_idx = df_price.index.get_loc(target_date)
    current_ai_capital = INITIAL_CAPITAL
    
    bench_capital = INITIAL_CAPITAL * (1 - ACTUAL_BUY_FEE) # 0050 首日扣除買入手續費
    
    previous_stocks = set() # 記錄上一期的持股名單
    milestones = []
    total_transaction_costs = 0

    # ==========================================
    # 執行滾動回測迴圈
    # ==========================================
    for period in range(1, NUM_PERIODS + 1):
        if current_idx >= len(df_price) - 1:
            print(f"\n⚠️ 資料已見底，無法進行第 {period} 期的回測。")
            break

        buy_date = df_price.index[current_idx]
        sell_idx = current_idx + HOLDING_DAYS
        
        if sell_idx >= len(df_price):
            print(f"\n⚠️ 警告：第 {period} 週期尚未滿 {HOLDING_DAYS} 個交易日，將使用最後一天結算。")
            sell_idx = len(df_price) - 1
            
        sell_date = df_price.index[sell_idx]
        actual_holding_days = sell_idx - current_idx

        if buy_date in df_pred.index:
            predictions_on_buy_date = df_pred.loc[buy_date]
            
            # 🔴 找出看漲的股票 (大於 0)
            positive_preds = predictions_on_buy_date[predictions_on_buy_date > 0]
            
            # 🔴 取交集：只留下「看漲」且「在品質白名單內」的股票
            safe_positive_preds = positive_preds[positive_preds.index.isin(qualified_symbols)]
            
        else:
            print(f"\n⚠️ 找不到 {buy_date.date()} 預測資料，滾動終止。")
            break

        print(f"\n" + "="*55)
        print(f"🔄 第 {period} 期投資 ({actual_holding_days} 個交易日)")
        print(f"📅 買入日: {buy_date.date()} | 📅 結算日: {sell_date.date()}")
        print("-" * 55)

        period_trade_cost = 0

        # 🔴 改用 safe_positive_preds 進行判斷
        if len(safe_positive_preds) == 0:
            print("當天沒有符合品質標準且看漲的股票，保持空手 (現金)。")
            current_stocks = set()
            portfolio_return = 0.0
        else:
            # 從優等生中挑選漲幅最高的 Top N
            top_n_stocks = safe_positive_preds.nlargest(TOP_N)
            current_stocks = set(top_n_stocks.index)
            # 維持等權重投資策略 (Equal-Weight)
            weight_per_stock = 1.0 / len(top_n_stocks)
            
            print(f"🎯 AI 最新預測看漲名單 (已通過品質審查):")
            for stock in current_stocks:
                print(f" - {stock}: (信心分數: {top_n_stocks[stock]:.2%})")

        # ==========================================
        # 計算換股邏輯與手續費
        # ==========================================
        sold_stocks = previous_stocks - current_stocks
        bought_stocks = current_stocks - previous_stocks
        kept_stocks = previous_stocks.intersection(current_stocks)
        
        old_stock_count = len(previous_stocks)
        new_stock_count = len(current_stocks)
        
        # 1. 計算賣出成本
        if old_stock_count > 0 and len(sold_stocks) > 0:
            sell_fraction = len(sold_stocks) / old_stock_count
            sell_value = current_ai_capital * sell_fraction
            sell_cost = sell_value * ACTUAL_SELL_FEE
            current_ai_capital -= sell_cost
            period_trade_cost += sell_cost
            
        # 2. 計算買入成本
        if new_stock_count > 0 and len(bought_stocks) > 0:
            buy_fraction = len(bought_stocks) / new_stock_count
            buy_value = current_ai_capital * buy_fraction
            buy_cost = buy_value * ACTUAL_BUY_FEE
            current_ai_capital -= buy_cost
            period_trade_cost += buy_cost
        elif old_stock_count == 0 and new_stock_count > 0:
            # 第一期買進全部
            buy_cost = current_ai_capital * ACTUAL_BUY_FEE
            current_ai_capital -= buy_cost
            period_trade_cost += buy_cost
            
        total_transaction_costs += period_trade_cost

        # 輸出換股資訊
        if period == 1:
            print(f"\n🛒 首度建倉！買入 {len(bought_stocks)} 檔股票。")
        else:
            print(f"\n🔁 換股狀況: 賣出 {len(sold_stocks)} 檔 | 保留 {len(kept_stocks)} 檔 | 新買 {len(bought_stocks)} 檔")
            
        print(f"💸 本期交易成本 (手續費+稅): {period_trade_cost:,.0f} 元")

        # 計算該期真實獲利 (僅針對持有的股票)
        if new_stock_count > 0:
            stock_returns = (df_price.loc[sell_date, list(current_stocks)] - df_price.loc[buy_date, list(current_stocks)]) / df_price.loc[buy_date, list(current_stocks)]
            portfolio_return = (stock_returns * weight_per_stock).sum()
        
        # 結算本期資產
        current_ai_capital = current_ai_capital * (1 + portfolio_return)

        # 計算同期 0050 累積表現
        bench_end_price = benchmark_0050.loc[benchmark_0050.index <= sell_date, 'Close'].iloc[-1]
        if isinstance(bench_end_price, pd.Series): bench_end_price = bench_end_price.item()
        bench_cum_return = (bench_end_price - bench_start_price) / bench_start_price
        
        # 0050 當前價值 (如果現在賣掉要扣稅費)
        bench_current_nav = bench_capital * (1 + bench_cum_return)
        bench_nav_after_tax = bench_current_nav * (1 - ACTUAL_SELL_FEE)

        print(f"📈 本期 AI 組合漲跌幅: {portfolio_return:+.2%}")
        print(f"💰 期末總資產: {current_ai_capital:,.0f} 元")

        milestones.append({
            'Period': period,
            'Total_Days': period * HOLDING_DAYS,
            'AI_Capital': current_ai_capital,
            'AI_Cum_Return': (current_ai_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'Bench_Capital': bench_nav_after_tax,
            'Bench_Cum_Return': (bench_nav_after_tax - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'Cost': period_trade_cost
        })

        previous_stocks = current_stocks
        current_idx = sell_idx

    # ==========================================
    # 輸出終極滾動比較總表
    # ==========================================
    if milestones:
        # 回測結束，AI 組合也要模擬全部賣出變現的最終淨值
        final_liquidation_cost = current_ai_capital * ACTUAL_SELL_FEE if len(previous_stocks) > 0 else 0
        final_ai_capital = current_ai_capital - final_liquidation_cost
        total_transaction_costs += final_liquidation_cost
        
        # 更新最後一期的紀錄
        milestones[-1]['AI_Capital'] = final_ai_capital
        milestones[-1]['AI_Cum_Return'] = (final_ai_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        print(f"\n\n{'='*70}")
        print(f"🏆 終極複利對決：xLSTM 滾動投資 vs 0050 (已扣除交易手續費與稅)")
        print(f"初始資金: {INITIAL_CAPITAL:,.0f} 元 | 總滾動次數: {NUM_PERIODS} 次")
        print(f"{'='*70}")
        print(f"{'期間':<8} | {'xLSTM 組合淨值 (累積報酬)':<22} | {'0050 ETF 淨值 (累積報酬)'}")
        print("-" * 70)
        
        for m in milestones:
            ai_str = f"{m['AI_Capital']:>10,.0f} 元 ({m['AI_Cum_Return']:>+7.2%})"
            bench_str = f"{m['Bench_Capital']:>10,.0f} 元 ({m['Bench_Cum_Return']:>+7.2%})"
            print(f"{m['Total_Days']:>3} 交易日 | {ai_str:<22} | {bench_str}")
        
        print(f"{'='*70}")
        print(f"⚠️ xLSTM 累計摩擦成本 (總手續費與稅金支出): {total_transaction_costs:,.0f} 元")
        
        diff = final_ai_capital - milestones[-1]['Bench_Capital']
        if diff > 0:
            print(f"🔥 最終結果：xLSTM 組合扣除成本後，大勝 0050 ETF 【 {diff:,.0f} 元 】！")
        else:
            print(f"📉 最終結果：xLSTM 組合扣除成本後，落後 0050 ETF 【 {-diff:,.0f} 元 】。")

if __name__ == "__main__":
    main()