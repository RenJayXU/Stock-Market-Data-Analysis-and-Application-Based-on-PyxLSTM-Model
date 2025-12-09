import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import ta # <-- 導入 ta

# 假設您要處理 2330.csv
STOCK_ID = "2330"

# 1. 讀取原始資料
df = pd.read_csv(f"{STOCK_ID}.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index(ascending=True)

# 2. 補齊所有工作日（週一到週五），用前一天資料填補
all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
df = df.reindex(all_dates)
df.ffill(inplace=True)

# 3. 【新增】 計算技術指標
print("計算技術指標 (RSI, MACD, SMA)...")
# 計算 RSI
df['RSI_14'] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
# 計算 MACD
df['MACD_12_26_9'] = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9).macd()
# 計算 SMA
df['SMA_10'] = ta.trend.SMAIndicator(close=df["Close"], window=10).sma_indicator()
df['SMA_20'] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()

# 由於指標計算初期會有 NaN，我們用 "bfill" (向後填充) 和 "ffill" (向前填充) 來補滿
df.bfill(inplace=True)
df.ffill(inplace=True)

# 4. 選擇需要的特徵 (現在總共有 9 個)
features = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD_12_26_9", "SMA_10", "SMA_20"
]
data = df[features].copy()

# 5. 按時間分割訓練集與測試集（80%訓練，20%測試）
split_date = data.index[int(len(data)*0.8)]
train_data = data.loc[data.index < split_date]
test_data = data.loc[data.index >= split_date]

# 6. 標準化 (StandardScaler)
scaler = StandardScaler()
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# 7. 儲存標準化後的資料
pd.DataFrame(train_scaled, columns=features, index=train_data.index).to_csv(f"train_{STOCK_ID}.csv")
pd.DataFrame(test_scaled, columns=features, index=test_data.index).to_csv(f"test_{STOCK_ID}.csv")

# 8. 儲存scaler物件
joblib.dump(scaler, f"{STOCK_ID}_scaler.save")

# 9. 輸出訓練集與測試集的基本統計資訊
print("\n=== 資料集統計 ===")
print(f"總特徵數: {len(features)}")
print(f"訓練集日期範圍: {train_data.index.min()} 至 {train_data.index.max()}")
print(f"訓練集樣本數: {len(train_data)}")
print(f"測試集日期範圍: {test_data.index.min()} 至 {test_data.index.max()}")
print(f"測試集樣本數: {len(test_data)}")

print(f"\n資料處理完成，已儲存 {STOCK_ID} 的檔案。")