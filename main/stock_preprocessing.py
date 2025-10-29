import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# 1. 讀取原始資料
df = pd.read_csv("3008.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index(ascending=True)

# 2. 補齊所有工作日（週一到週五），用前一天資料填補
all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
df = df.reindex(all_dates)
df.ffill(inplace=True)  # 用前一天資料填補

# 3. 檢查資料連續性 - 修改為更合理的檢查
# 工作日之間的差值可能是1、3（週末）或更多（假日），但不應有缺失的工作日
business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
missing_days = set(business_days) - set(df.index)
if missing_days:
    print(f"警告：仍有 {len(missing_days)} 個工作日缺失")
    print(f"缺失日期範例：{list(missing_days)[:5]}")
else:
    print("✓ 所有工作日都已補齊")

# 4. 選擇需要的特徵
features = ["Open", "High", "Low", "Close", "Volume"]
data = df[features].copy()

# 5. 按時間分割訓練集與測試集（80%訓練，20%測試，確保測試集在訓練集之後）
split_date = data.index[int(len(data)*0.8)]
train_data = data.loc[data.index < split_date]
test_data = data.loc[data.index >= split_date]

# 6. 標準化（只用訓練集fit，再transform兩組資料）
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# 7. 儲存標準化後的資料
pd.DataFrame(train_scaled, columns=features, index=train_data.index).to_csv("train_3008.csv")
pd.DataFrame(test_scaled, columns=features, index=test_data.index).to_csv("test_3008.csv")

# 8. 儲存scaler物件
joblib.dump(scaler, "3008_scaler.save")
# 9. 輸出訓練集與測試集的基本統計資訊
print("\n=== 資料集統計 ===")
print(f"訓練集日期範圍: {train_data.index.min()} 至 {train_data.index.max()}")
print(f"訓練集樣本數: {len(train_data)}")
print(f"測試集日期範圍: {test_data.index.min()} 至 {test_data.index.max()}")
print(f"測試集樣本數: {len(test_data)}")

print("\n資料補齊、填補與標準化完成，訓練集與測試集已分別儲存。")


