# metrics.py
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, y_prev_true):
    """
    y_true: 當天實際收盤價
    y_pred: 當天預測收盤價
    y_prev_true: 前一天實際收盤價 (用於計算趨勢)
    """
    # 1. 基礎誤差指標
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 2. R2 Score (防呆)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 1e-10 else float('nan')

    # 3. 趨勢指標 (Trend Accuracy)
    # 邏輯：(今天預測 - 昨天實際) vs (今天實際 - 昨天實際)
    real_movement = np.sign(y_true - y_prev_true)
    pred_movement = np.sign(y_pred - y_prev_true)
    
    # 處理 0 的情況 (不變也視為一種狀態，或歸類為跌，這邊簡化為 sign 比較)
    acc = accuracy_score(real_movement, pred_movement)
    f1 = f1_score(real_movement, pred_movement, average='weighted', zero_division=0)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": acc,
        "F1": f1
    }