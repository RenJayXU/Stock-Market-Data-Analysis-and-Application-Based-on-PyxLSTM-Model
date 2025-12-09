import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM

# 定義要預測的股票清單
STOCK_LIST = ['1301', '2330', '2882', '1734', '3008']

# 設定參數 (必須與訓練時一致)
CONFIG = {
    "input_size": 9,
    "hidden_size": 128,
    "num_layers": 2,
    "num_blocks": 2,
    "dropout": 0.3,
    "sequence_length": 30
}

def safe_r2_score(y_true, y_pred):
    """計算 R2 Score，防止分母為零"""
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10: 
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_stock(stock_id):
    print(f"\n{'='*10} 正在評估股票: {stock_id} {'='*10}")
    
    # 1. 檢查並載入 Scaler
    scaler_path = f"processed_data/{stock_id}_scaler.save"
    if not os.path.exists(scaler_path):
        print(f"[錯誤] 找不到 scaler 檔案: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    
    # 2. 準備測試資料
    # 我們同時需要 DataLoader (給模型用) 和原始 DataFrame (為了取得正確日期)
    test_file = f"processed_data/test_{stock_id}.csv"
    if not os.path.exists(test_file):
        print(f"[錯誤] 找不到測試資料: {test_file}")
        return
        
    # 讀取 CSV 以取得日期索引
    df_test = pd.read_csv(test_file, index_col=0, parse_dates=True)
    
    # 建立 Dataset 與 DataLoader
    test_dataset = StockDataset(test_file, sequence_length=CONFIG["sequence_length"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    # 3. 載入模型
    model = StockxLSTM(
        input_size=CONFIG["input_size"], 
        hidden_size=CONFIG["hidden_size"], 
        num_layers=CONFIG["num_layers"], 
        num_blocks=CONFIG["num_blocks"], 
        dropout=CONFIG["dropout"],
        lstm_type="slstm"
    )
    
    model_path = f"models/{stock_id}_best_model.pth"
    if not os.path.exists(model_path):
        print(f"[錯誤] 找不到模型檔案: {model_path}，請先執行 stock_train.py")
        return
        
    # 載入權重 (映射到 CPU 以免報錯)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. 執行預測
    predictions = []   # 預測值 (Scaled)
    actuals = []       # 實際值 (Scaled)
    prev_actuals = []  # 前一天的實際值 (Scaled，用於計算漲跌)
    
    print("正在執行預測...")
    with torch.no_grad():
        for input_seq, target in test_loader:
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            # input_seq shape: (1, 30, 9)
            pred = model(input_seq)
            
            predictions.append(pred[0].cpu().numpy()) 
            actuals.append(target[0].cpu().numpy())
            
            # input_seq 的最後一個時間點 (index -1) 就是 "昨天" 的數據
            prev_actuals.append(input_seq[0, -1, :].cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    prev_actuals = np.array(prev_actuals)

    # 5. 反標準化 (只取 Index 3: Close Price)
    # 由於 Scaler 是針對 9 個特徵訓練的，我們需要用完整的 shape 來反轉
    def inverse_transform_close(data_array):
        # data_array shape: (N, 9)
        real_data = scaler.inverse_transform(data_array)
        return real_data[:, 3] # 只回傳 Close 欄位

    pred_close = inverse_transform_close(predictions)
    actual_close = inverse_transform_close(actuals)
    prev_close = inverse_transform_close(prev_actuals)

    # 6. 計算評估指標
    mse = np.mean((pred_close - actual_close) ** 2)
    mae = np.mean(np.abs(pred_close - actual_close))
    r2 = safe_r2_score(actual_close, pred_close)
    
    # 【關鍵修正】漲跌趨勢準確度計算
    # 實際趨勢 = 今天實際收盤 - 昨天實際收盤
    real_trend = (actual_close - prev_close > 0).astype(int)
    # 預測趨勢 = 今天預測收盤 - 昨天實際收盤 (這才是交易邏輯)
    pred_trend = (pred_close - prev_close > 0).astype(int)
    
    acc = accuracy_score(real_trend, pred_trend)
    f1 = f1_score(real_trend, pred_trend, zero_division=0)

    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print(f"Trend Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # 7. 儲存結果供投資組合使用 (Portfolio Optimization)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 處理日期對齊
    # Dataset 的第 i 筆資料，對應的是原始 DataFrame 中第 (Sequence_length + i) 筆資料的 Target
    # 所以我們從第 30 筆資料開始取日期
    pred_dates = df_test.index[CONFIG["sequence_length"] : CONFIG["sequence_length"] + len(pred_close)]
    
    # 確保長度一致 (防呆)
    if len(pred_dates) != len(pred_close):
        print(f"[警告] 日期長度 ({len(pred_dates)}) 與 預測長度 ({len(pred_close)}) 不符，截斷處理。")
        min_len = min(len(pred_dates), len(pred_close))
        pred_dates = pred_dates[:min_len]
        pred_close = pred_close[:min_len]

    # 建立 DataFrame 並存檔
    prediction_df = pd.DataFrame({
        "Date": pred_dates,
        "Predicted_Close": pred_close
    })
    
    # 存成 CSV，這就是 portfolio_optimization_final.py 要讀取的檔案
    save_csv_path = f"{results_dir}/future_prediction_{stock_id}.csv"
    prediction_df.to_csv(save_csv_path, index=False)
    print(f"已儲存預測結果至: {save_csv_path}")

    # 8. 寫入評估報告與畫圖
    with open(f"{results_dir}/{stock_id}_performance.txt", "w") as f:
        f.write(f"Stock: {stock_id}\n")
        f.write(f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nR2: {r2:.6f}\n")
        f.write(f"Accuracy: {acc:.6f}\nF1: {f1:.6f}\n")

    # 畫圖
    plt.figure(figsize=(12, 8))
    
    # 子圖 1: 價格走勢
    plt.subplot(2, 1, 1)
    plt.plot(pred_dates, actual_close, label='Actual Price', alpha=0.7)
    plt.plot(pred_dates, pred_close, label='Predicted Price', alpha=0.7, linestyle='--')
    plt.title(f"{stock_id} Price Prediction (xLSTM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖 2: 趨勢預測 (前 100 點)
    plt.subplot(2, 1, 2)
    limit = min(100, len(real_trend))
    plt.plot(range(limit), real_trend[:limit], 'g-', label='Real Trend', alpha=0.6)
    plt.plot(range(limit), pred_trend[:limit], 'r--', label='Pred Trend', alpha=0.6)
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.title("Trend Prediction (First 100 days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{stock_id}_prediction.png")
    plt.close()

if __name__ == "__main__":
    # 確保結果資料夾存在
    os.makedirs("results", exist_ok=True)
    
    for stock in STOCK_LIST:
        evaluate_stock(stock)
        
    print("\n所有股票預測完成！請繼續執行 portfolio_optimization_final.py")