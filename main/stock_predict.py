import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, accuracy_score
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM

STOCK_ID = "2330"

def safe_r2_score(y_true, y_pred):
    """安全計算R²，避免除零錯誤"""
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_trend_metrics(actuals, preds):
    """計算漲跌分類指標"""
    # 生成漲跌標籤（1:漲, 0:跌）
    actual_labels = (np.diff(actuals) > 0).astype(int)
    pred_labels = (np.diff(preds) > 0).astype(int)
    
    # 處理長度差異
    min_length = min(len(actual_labels), len(pred_labels))
    actual_labels = actual_labels[:min_length]
    pred_labels = pred_labels[:min_length]
    
    # 計算指標
    f1 = f1_score(actual_labels, pred_labels)
    acc = accuracy_score(actual_labels, pred_labels)
    
    return f1, acc, actual_labels, pred_labels

def main():
    # 1. 載入訓練時儲存的scaler
    scaler = joblib.load(f"{STOCK_ID}_scaler.save")
    
    # 2. 準備測試集
    test_dataset = StockDataset(f"test_{STOCK_ID}.csv", sequence_length=30)    
    test_size = min(100, len(test_dataset))
    test_indices = range(len(test_dataset) - test_size, len(test_dataset))
        
    # 3. 載入模型
    model = StockxLSTM(
        input_size=9, 
        hidden_size=128,  # <-- 保持 256
        num_layers=2, 
        num_blocks=2, 
        dropout=0.4,
        lstm_type="slstm" # <-- 【修改】
    )
    model.load_state_dict(torch.load(f"{STOCK_ID}model.pth"))
    model.eval()

    # 4. 預測
    predictions = []
    actuals = []
    with torch.no_grad():
        for i in test_indices:
            input_seq, target = test_dataset[i]
            input_seq = input_seq.unsqueeze(0)
            pred = model(input_seq)
            predictions.append(pred[0].detach().numpy())
            actuals.append(target.numpy())

    # 轉換為NumPy陣列
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 5. 反標準化
    dummy_pred = np.zeros_like(predictions)
    dummy_pred[:, :] = predictions
    predictions_real = scaler.inverse_transform(dummy_pred)[:, 3]

    dummy_actual = np.zeros_like(actuals)
    dummy_actual[:, :] = actuals
    actuals_real = scaler.inverse_transform(dummy_actual)[:, 3]

    # 6. 數據驗證
    print("\n=== 數據驗證 ===")
    print(f"實際收盤價範圍: {actuals_real.min():.2f} - {actuals_real.max():.2f}")
    print(f"預測收盤價範圍: {predictions_real.min():.2f} - {predictions_real.max():.2f}")

    # 7. 計算評估指標
    mse = np.mean((predictions_real - actuals_real) ** 2)
    mae = np.mean(np.abs(predictions_real - actuals_real))
    r2 = safe_r2_score(actuals_real, predictions_real)
    
    # 新增漲跌分類指標
    f1, acc, actual_labels, pred_labels = calculate_trend_metrics(actuals_real, predictions_real)

    print("\n=== 預測性能評估 ===")
    print(f"MSE (均方誤差): {mse:.6f}")
    print(f"MAE (平均絕對誤差): {mae:.6f}")
    if not np.isnan(r2):
        print(f"R² (決定係數): {r2:.4f}")
    else:
        print("R² (決定係數): 無效（實際值缺乏變化）")
    print(f"F1 Score (漲跌預測): {f1:.4f}")
    print(f"Accuracy (漲跌預測): {acc:.4f}")

    # 8. 可視化結果
    plt.figure(figsize=(18, 9))
    
    # 主圖：價格走勢
    plt.subplot(2, 1, 1)
    plt.plot(actuals_real, label='Actual closing price', linewidth=2, marker='o', markersize=5)
    plt.plot(predictions_real, label='Predicted closing price', linewidth=2, linestyle='--', marker='x', markersize=5)
    plt.title(f"{STOCK_ID} Price Forecast Results", fontsize=16)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 子圖：漲跌預測
    plt.subplot(2, 1, 2)
    plt.plot(actual_labels, 'g-', label='Actual Trend')
    plt.plot(pred_labels, 'r--', label='Predicted Trend')
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.title('Trend Prediction Accuracy', fontsize=14)
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Trend', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{STOCK_ID}stock_prediction_results.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
