import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import ta
import os
import warnings
from tqdm import tqdm

STOCK_LIST = [
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

def preprocess_stock(stock_id):
    file_path = f"data/{stock_id}.csv" 
    if not os.path.exists(file_path):
        file_path = f"{stock_id}.csv"
        
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到 {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index(ascending=True)
    df.dropna(inplace=True)

    # ==========================================
    # 🔴 核心修改 1：計算未來 20 天累積報酬率，並明確命名為目標
    # ==========================================
    # 代表「這天買進，持有 20 個交易日後的累積報酬率」
    df['Actual_Return'] = df['Close'].pct_change(20).shift(-20)
    
    # 2. 自動生成所有技術指標
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

    # 3. 過濾與選取特徵
    cols = df.columns.tolist()
    if 'Actual_Return' in cols:
        cols.remove('Actual_Return')
        
    # 移除基礎價量欄位，避免重複
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in cols:
            cols.remove(c)
            
    # 選擇所有技術指標
    selected_ta = cols 
    
    # 🔴 修正：最終特徵只包含基礎價量與 TA，絕對不包含目標值！
    features = ["Open", "High", "Low", "Close", "Volume"] + selected_ta
    
    # 把特徵和目標值組裝在一起
    data = df[features + ['Actual_Return']].copy()

    # 清除極端無效值以及 shift(-20) 導致最後 20 天產生的 NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # ==========================================
    # 嚴格基於時間切割資料集
    # ==========================================
    data.index = pd.to_datetime(data.index)
    
    train_data = data.loc['2015-01-01':'2023-12-31']
    # 為了讓 2024 年和 2025 年的預測有 120 天的助跑期，往前推半年切分
    val_data = data.loc['2023-07-01':'2024-12-31']
    test_data = data.loc['2024-07-01':]

    if train_data.empty or val_data.empty or test_data.empty:
        print(f"\n⚠️ 警告：{stock_id} 資料切分有空缺。這通常發生在新上市的股票。")
        if train_data.empty: return None

    # ==========================================
    # 🔴 核心修改 2：只對特徵 (features) 進行標準化，保留目標的真實數值
    # ==========================================
    scaler = StandardScaler() 
    # 拿 Train 的特徵去 fit
    scaler.fit(train_data[features])
    
    # 對特徵進行 transform
    train_scaled_features = scaler.transform(train_data[features])
    val_scaled_features = scaler.transform(val_data[features])
    test_scaled_features = scaler.transform(test_data[features])
    
    # 把它跟沒有被標準化的 'Actual_Return' 合併回去
    train_final = pd.DataFrame(train_scaled_features, columns=features, index=train_data.index)
    train_final['Actual_Return'] = train_data['Actual_Return'].values
    
    val_final = pd.DataFrame(val_scaled_features, columns=features, index=val_data.index)
    val_final['Actual_Return'] = val_data['Actual_Return'].values
    
    test_final = pd.DataFrame(test_scaled_features, columns=features, index=test_data.index)
    test_final['Actual_Return'] = test_data['Actual_Return'].values

    # 6. 儲存處理後的資料
    os.makedirs("processed_data", exist_ok=True)
    
    train_final.to_csv(f"processed_data/train_{stock_id}.csv")
    val_final.to_csv(f"processed_data/val_{stock_id}.csv")
    test_final.to_csv(f"processed_data/test_{stock_id}.csv")

    joblib.dump(scaler, f"processed_data/{stock_id}_scaler.save")
    
    return len(features)

if __name__ == "__main__":
    print("開始進行前處理...")
    feature_count = 0
    for stock in tqdm(STOCK_LIST, desc="前處理進度"):
        count = preprocess_stock(stock)
        if count is not None:
            feature_count = count
            
    print(f"\n✅ 前處理完成！目前特徵總數量為：【 {feature_count} 】")