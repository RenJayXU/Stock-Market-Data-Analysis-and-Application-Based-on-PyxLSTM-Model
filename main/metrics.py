import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, y_prev=None):
    """
    計算回歸與分類指標
    y_true: 真實報酬率 (Actual Return)
    y_pred: 預測報酬率 (Predicted Return)
    y_prev: 已經不需要了，但為了相容 stock_predict.py 的呼叫而保留參數位置
    """
    # 1. 回歸指標
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (避免除以 0)
    # 注意：在報酬率預測中，真實值非常容易趨近於 0，MAPE 的數字可能會飆得非常大，這是正常的數學現象。
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    r2 = r2_score(y_true, y_pred)

    # ==========================================
    # 🔴 核心修復：方向性指標 (Directional Accuracy)
    # ==========================================
    # 因為輸入已經是「報酬率」，大於 0 就是賺錢(漲)，小於等於 0 就是賠錢(跌)
    true_dir = y_true > 0
    pred_dir = y_pred > 0
    
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