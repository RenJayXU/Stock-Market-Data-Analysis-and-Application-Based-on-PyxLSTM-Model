import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import ta
import os

# 定義要處理的股票清單
STOCK_LIST = ['1301', '2330', '2882', '1734', '3008']

def preprocess_stock(stock_id):
    print(f"\n=== 正在處理股票代碼: {stock_id} ===")
    
    # 1. 讀取原始資料 (假設 csv 在 data 資料夾下，或與腳本同目錄)
    # 請根據您實際的檔案路徑調整
    file_path = f"data/{stock_id}.csv" 
    if not os.path.exists(file_path):
        file_path = f"{stock_id}.csv"
        
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到 {file_path}")
        return

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index(ascending=True)

    # 2. 【修正】移除強制補齊工作日 (reindex 'B') 與 ffill
    # 我們只保留實際有交易的日期，避免在假日填入重複價格導致模型學習到錯誤的"零波動"特徵。
    # 檢查是否有 NaN 並移除
    df.dropna(inplace=True)

    # 3. 計算技術指標
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    # MACD
    df['MACD_12_26_9'] = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9).macd()
    # SMA
    df['SMA_10'] = ta.trend.SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df['SMA_20'] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()

    # 清除計算指標產生的 NaN
    df.dropna(inplace=True)

    # 4. 選擇特徵
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI_14", "MACD_12_26_9", "SMA_10", "SMA_20"
    ]
    data = df[features].copy()

    # 5. 【修正】分割資料集：訓練(70%)、驗證(10%)、測試(20%)
    # 驗證集用於 Early Stopping，測試集用於最終績效評估，完全隔離以防 Data Leakage
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    # 6. 標準化 
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    scaler.fit(train_data)
    
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 7. 儲存處理後的資料
    # 建議建立一個 processed_data 資料夾來存放
    os.makedirs("processed_data", exist_ok=True)
    
    pd.DataFrame(train_scaled, columns=features, index=train_data.index).to_csv(f"processed_data/train_{stock_id}.csv")
    pd.DataFrame(val_scaled, columns=features, index=val_data.index).to_csv(f"processed_data/val_{stock_id}.csv")
    pd.DataFrame(test_scaled, columns=features, index=test_data.index).to_csv(f"processed_data/test_{stock_id}.csv")

    # 儲存 Scaler 供預測時使用
    joblib.dump(scaler, f"processed_data/{stock_id}_scaler.save")
    
    print(f"完成 {stock_id}:")
    print(f"  - 訓練集: {len(train_data)} 筆 ({train_data.index.min().date()} ~ {train_data.index.max().date()})")
    print(f"  - 驗證集: {len(val_data)} 筆")
    print(f"  - 測試集: {len(test_data)} 筆")

if __name__ == "__main__":
    for stock in STOCK_LIST:
        preprocess_stock(stock)