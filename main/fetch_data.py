import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

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

def fetch_stock_data(stock_id):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{stock_id}.csv"
    
    # 🔴 獲取今天的日期，確保資料抓到最新，讓 shift(-20) 有未來的資料可以算
    today = datetime.today().strftime('%Y-%m-%d')
    
    try:
        # 先嘗試上市代碼
        ticker = f"{stock_id}.TW"
        # 🔴 將 end 設為 today，不再限制於 2025-01-31
        df = yf.download(ticker, start="2015-01-01", end=today, progress=False)
        
        # 若抓不到，改嘗試上櫃代碼
        if df.empty:
            ticker = f"{stock_id}.TWO"
            df = yf.download(ticker, start="2015-01-01", end=today, progress=False)
            
        if not df.empty:
            # 處理 yfinance 新版可能產生的 MultiIndex 欄位問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # 篩選並確保欄位與原本的格式一致
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.set_index('Date', inplace=True)
            
            # 儲存到 data 資料夾
            df.to_csv(file_path)
        else:
            print(f"\n警告: 無法取得 {stock_id} 的資料")
            
    except Exception as e:
        print(f"\n下載 {stock_id} 時發生錯誤: {e}")

if __name__ == "__main__":
    print("開始下載成分股最新歷史資料...")
    for stock in tqdm(STOCK_LIST, desc="資料下載進度"):
        fetch_stock_data(stock)
    print("✅ 資料下載完成！請接著執行 stock_preprocessing.py")