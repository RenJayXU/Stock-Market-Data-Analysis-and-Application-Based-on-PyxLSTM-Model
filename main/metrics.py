import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, y_prev):
    """
    計算回歸與分類指標
    y_true: 真實股價
    y_pred: 預測股價
    y_prev: 前一日股價 (用於計算漲跌趨勢)
    """
    # 1. 回歸指標
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (避免除以 0)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    r2 = r2_score(y_true, y_pred)

    # 2. 方向性指標 (Directional Accuracy)
    # 真實漲跌: (今天 - 昨天) > 0 為 1 (漲), 否則 0 (跌)
    true_dir = (y_true - y_prev) > 0
    pred_dir = (y_pred - y_prev) > 0
    
    accuracy = accuracy_score(true_dir, pred_dir)
    f1 = f1_score(true_dir, pred_dir, zero_division=0)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Accuracy": accuracy,
        "F1": f1
    }