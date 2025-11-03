import torch
import pandas as pd
import numpy as np
import joblib
from stock_xlstm import StockxLSTM
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import ta # <-- 導入 ta

def predict_future_close(num_days=60):
    
    # --- 1. 設定 & 載入 ---
    STOCK_ID = "3008" # 確保這裡的代號與您訓練的一致
    
    # 檢測 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    scaler = joblib.load(f"{STOCK_ID}_scaler.save")
    rawdata = pd.read_csv(f"{STOCK_ID}.csv", parse_dates=["Date"], index_col="Date")
    rawdata = rawdata.sort_index(ascending=True)
    last_date = rawdata.index[-1]

    # --- 2. 特徵工程 ---
    # 【關鍵修正】 使用 'ta' 函式庫的正確語法
    print("正在為歷史資料計算技術指標...")
    
    # (錯誤的 'pandas-ta' 語法)
    # rawdata.ta.rsi(length=14, append=True)
    # rawdata.ta.macd(fast=12, slow=26, signal=9, append=True)
    # rawdata.ta.sma(length=10, append=True)
    # rawdata.ta.sma(length=20, append=True)
    
    # (正確的 'ta' 語法)
    rawdata['RSI_14'] = ta.momentum.RSIIndicator(close=rawdata["Close"], window=14).rsi()
    rawdata['MACD_12_26_9'] = ta.trend.MACD(close=rawdata["Close"], window_slow=26, window_fast=12, window_sign=9).macd()
    rawdata['SMA_10'] = ta.trend.SMAIndicator(close=rawdata["Close"], window=10).sma_indicator()
    rawdata['SMA_20'] = ta.trend.SMAIndicator(close=rawdata["Close"], window=20).sma_indicator()
    
    # 清理 ta 產生的 inf 和 nan
    rawdata.replace([np.inf, -np.inf], np.nan, inplace=True)
    rawdata.bfill(inplace=True) # 用後面的值往前填
    rawdata.ffill(inplace=True) # 用前面的值往後填

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI_14", "MACD_12_26_9", "SMA_10", "SMA_20"
    ]
    
    featuredata = rawdata[features]
    processeddata = scaler.transform(featuredata) # 使用載入的 scaler 轉換

    # --- 3. 建立模型 (參數必須與訓練時完全一致) ---
    model = StockxLSTM(
        input_size=9,
        hidden_size=128,
        num_layers=2,
        num_blocks=2,
        dropout=0.4, 
        lstm_type="slstm" # 使用 sLSTM
    )
    model.load_state_dict(torch.load(f"{STOCK_ID}model.pth"))
    model.to(device) 
    model.eval()

    # --- 4. 迴歸預測 ---
    input_sequence = processeddata[-30:].copy() # shape (30, 9)
    predictions = []
    current_date = last_date

    print(f"開始進行未來 {num_days} 天的預測...")
    with torch.no_grad():
        for i in range(num_days):
            # 準備輸入
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.to(device) 
            
            # 進行預測 (pred_scaled)
            pred_scaled = model(input_tensor).cpu().numpy()[0]  # shape: (9,)
            
            # 反標準化 (Inverse Transform)
            dummy = np.zeros((1, 9))
            dummy[0, :] = pred_scaled
            real_pred_all_features = scaler.inverse_transform(dummy)[0]
            
            # "Close" 是第 3 號索引
            close_pred_real = real_pred_all_features[3]

            # 儲存結果
            pred_date = (current_date + BDay(1)).strftime('%Y-%m-%d')
            predictions.append((pred_date, close_pred_real))
            current_date += BDay(1)

            # 更新輸入序列
            new_day_features_scaled = pred_scaled
            input_sequence = np.vstack([input_sequence[1:], new_day_features_scaled])

    print("\n未來{}天Close預測：".format(num_days))
    print("=================================")
    print("Date        Close")
    for date, close in predictions:
        print(f"{date}  {close:8.2f}")
    print("=================================")

    # --- 5. 儲存與繪圖 ---
    pred_dates = [x[0] for x in predictions]
    close_prices = [x[1] for x in predictions]
    plt.figure(figsize=(10, 6))
    plt.plot(pred_dates, close_prices, marker='o', linestyle='-', color='b')
    plt.title(f'Predicted Close Price ({STOCK_ID}) for Next {num_days} Days (sLSTM)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Predicted_Close_Price_{STOCK_ID}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    df = pd.DataFrame(predictions, columns=["Date", "Predicted_Close"])
    df.to_excel(f"future_prediction{STOCK_ID}.xlsx", index=False)
    print(f"已儲存預測為 future_prediction{STOCK_ID}.xlsx")

if __name__ == "__main__":
    predict_future_close(num_days=60)