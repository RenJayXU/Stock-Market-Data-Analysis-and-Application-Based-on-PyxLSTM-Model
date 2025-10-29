import pandas as pd
import numpy as np
import joblib
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from stock_predict import safe_r2_score, calculate_trend_metrics

def train_arima(
    train_csv_path='train_3008.csv',
    test_csv_path='test_3008.csv',
    scaler_path='3008_scaler.save',
    target_col=3  # Close在标准化后的第4列(0-based)
):
    # 加载scaler
    scaler = joblib.load(scaler_path)
    
    # 读取训练集与测试集
    train_data = pd.read_csv(train_csv_path, index_col=0)
    test_data = pd.read_csv(test_csv_path, index_col=0)
    
    # 提取Close列(标准化后)
    train_close = train_data.iloc[:, target_col].values
    test_close = test_data.iloc[:, target_col].values

    # 改善：自動搜尋p,d,q，允許更高階差分與更廣泛參數
    model = auto_arima(
        train_close,
        seasonal=False,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        max_d=3,
        test='adf',  # 單位根檢驗
        information_criterion='aic',
        trace=True,
        stepwise=False,  # 全面搜尋
        error_action='ignore',
        suppress_warnings=True
    )
    
    # 预测测试集
    n_test = len(test_close)
    predictions = model.predict(n_test)
    
    # 反标准化处理
    dummy_pred = np.zeros((n_test, 5))
    dummy_pred[:, target_col] = predictions
    pred_close = scaler.inverse_transform(dummy_pred)[:, target_col]
    
    # 实际值反标准化
    dummy_actual = np.zeros((n_test, 5))
    dummy_actual[:, target_col] = test_close
    actual_close = scaler.inverse_transform(dummy_actual)[:, target_col]
    
    # 计算指标
    mse = mean_squared_error(actual_close, pred_close)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = safe_r2_score(actual_close, pred_close)
    
    # 趋势指标计算
    f1, acc, actual_labels, pred_labels = calculate_trend_metrics(actual_close, pred_close)
    
    # 打印结果
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    return pred_close, actual_close

if __name__ == '__main__':
    preds, actuals = train_arima()
