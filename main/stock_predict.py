import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from torch.utils.data import DataLoader

# 引入自定義模組
from stock_xlstm import StockxLSTM
from stock_dataset import StockDataset
# 【新增】引入指標計算功能
from metrics import calculate_metrics

# 設定參數 (必須與 stock_train.py 完全一致)
CONFIG = {
    "sequence_length": 30,
    "prediction_days": 1,
    "hidden_size": 128,      # 配合訓練參數
    "num_layers": 1,         # 配合訓練參數
    "num_blocks": 2,         # 配合訓練參數
    "dropout": 0.3,
    "lstm_type": ['mlstm', 'slstm'] # 配合混合架構
}

STOCK_LIST = ['1301', '2330', '2882', '1734', '3008']

def predict_stock(stock_id, device):
    print(f"\n=== 正在預測股票: {stock_id} ===")
    
    # 1. 載入 Scaler (用於反正規化)
    scaler_path = f"processed_data/{stock_id}_scaler.save"
    if not os.path.exists(scaler_path):
        print(f"錯誤: 找不到 Scaler {scaler_path}")
        return
    scaler = joblib.load(scaler_path)

    # 2. 準備測試資料集
    test_csv = f"processed_data/test_{stock_id}.csv"
    if not os.path.exists(test_csv):
        print(f"錯誤: 找不到測試資料 {test_csv}")
        return
        
    # 讀取原始日期索引
    df_test_raw = pd.read_csv(test_csv, index_col=0)
    test_dates = pd.to_datetime(df_test_raw.index)

    # 建立 Dataset 與 DataLoader
    test_dataset = StockDataset(
        test_csv,
        sequence_length=CONFIG["sequence_length"],
        prediction_days=CONFIG["prediction_days"]
    )
    
    # Batch size 設為 1 以便模擬逐日預測
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 載入模型
    model = StockxLSTM(
        input_size=9,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_blocks=CONFIG["num_blocks"],
        dropout=CONFIG["dropout"],
        lstm_type=CONFIG["lstm_type"]
    )
    
    model_path = f"models/{stock_id}_best_model.pth"
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型權重 {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型權重載入成功！")
    except RuntimeError as e:
        print(f"模型載入失敗，請檢查參數是否與訓練時一致。\n錯誤訊息: {e}")
        return

    model.to(device)
    model.eval()

    # 4. 執行預測
    predictions = []
    actuals = []
    prev_actuals = [] # 【新增】用於儲存前一日價格 (計算漲跌幅用)
    
    print("開始推論...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            
            # 儲存預測值與真實值
            predictions.append(output.cpu().numpy()[0])
            actuals.append(targets.numpy()[0])
            
            # 【新增】儲存輸入序列的最後一天 (即預測日的前一天)
            # inputs shape: (1, seq_len, features) -> 取最後一個時間點
            prev_actuals.append(inputs[0, -1, :].cpu().numpy())

    predictions = np.array(predictions) # Shape: (N, 9)
    actuals = np.array(actuals)         # Shape: (N, 9)
    prev_actuals = np.array(prev_actuals) # Shape: (N, 9)

    # 5. 反正規化 (Inverse Transform)
    pred_inverse = scaler.inverse_transform(predictions)
    actual_inverse = scaler.inverse_transform(actuals)
    prev_inverse = scaler.inverse_transform(prev_actuals)

    # 取出 Close Price (index 3)
    pred_close = pred_inverse[:, 3]
    actual_close = actual_inverse[:, 3]
    prev_close = prev_inverse[:, 3]

    # 對齊日期
    valid_length = len(pred_close)
    plot_dates = test_dates[-valid_length:]

    # 6. 【關鍵修改】計算指標並儲存報告 (給 generate_report.py 使用)
    metrics = calculate_metrics(actual_close, pred_close, prev_close)
    
    print(f"[{stock_id}] 測試集表現:")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  Accuracy (漲跌準確度): {metrics['Accuracy']:.2%}")

    # 儲存 Metrics 到 txt
    os.makedirs("results", exist_ok=True)
    with open(f"results/{stock_id}_performance.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"評估報告已儲存至 results/{stock_id}_performance.txt")

    # 7. 儲存預測結果 CSV 與繪圖
    result_df = pd.DataFrame({
        "Date": plot_dates,
        "Actual_Close": actual_close,
        "Predicted_Close": pred_close
    })
    result_df.to_csv(f"results/future_prediction_{stock_id}.csv", index=False)
    
    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, actual_close, label="Actual Price", color='black', alpha=0.6)
    plt.plot(plot_dates, pred_close, label="Predicted Price (xLSTM)", color='#E24A33', alpha=0.9, linewidth=1.5)
    plt.title(f"{stock_id} Price Prediction (xLSTM Hybrid)\nAcc: {metrics['Accuracy']:.2%}, MSE: {metrics['MSE']:.5f}")
    plt.xlabel("Date")
    plt.ylabel("Price (TWD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/{stock_id}_prediction.png")
    plt.close()
    print("預測圖表已儲存。")

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for stock in STOCK_LIST:
        predict_stock(stock, device)

if __name__ == "__main__":
    main()