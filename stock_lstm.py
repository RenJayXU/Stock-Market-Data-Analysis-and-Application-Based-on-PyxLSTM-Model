import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
import os
import sys
import time

# 將上層目錄加入 path 以便讀取 stock_dataset 和 metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_dataset import StockDataset
try:
    from metrics import calculate_metrics
except ImportError:
    print("請確認 metrics.py 是否存在於 main 資料夾中")
    exit()

# 設定參數 (保持與 xLSTM 一致)
CONFIG = {
    "stock_list": ['1301', '2330', '2882', '1734', '3008'],
    "input_size": 9,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "sequence_length": 30,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_epochs": 200,
    "patience": 20
}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 9) # 輸出 9 個特徵

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最後一個時間點的輸出
        return self.fc(out[:, -1, :])

def run_lstm(stock_id, device):
    print(f"\n=== 執行 LSTM 模型: {stock_id} ===")
    
    # 1. 讀取資料
    train_dataset = StockDataset(f"processed_data/train_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    val_dataset = StockDataset(f"processed_data/val_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    test_dataset = StockDataset(f"processed_data/test_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 建立模型
    model = LSTMModel(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    # 3. 訓練與早停
    best_val_loss = float('inf')
    wait = 0
    save_path = f"models/{stock_id}_lstm.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 驗證 (只看 Close Price Loss 以對齊標準)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs[:, 3], targets[:, 3])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 4. 測試與評估
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(f"processed_data/{stock_id}_scaler.save")
    
    predictions, actuals, prev_actuals = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            # 預測
            pred = model(inputs)
            predictions.append(pred[0].cpu().numpy())
            actuals.append(targets[0].numpy())
            # 取得前一天實際值 (input_seq 的最後一筆)
            prev_actuals.append(inputs[0, -1, :].cpu().numpy())

    # 反標準化
    def inverse_close(data):
        return scaler.inverse_transform(np.array(data))[:, 3]

    pred_close = inverse_close(predictions)
    actual_close = inverse_close(actuals)
    prev_close = inverse_close(prev_actuals)

    # 計算指標
    metrics = calculate_metrics(actual_close, pred_close, prev_close)
    print(f"Result {stock_id}: {metrics}")
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for stock in CONFIG['stock_list']:
        m = run_lstm(stock, device)
        results.append({**{'Stock': stock, 'Model': 'LSTM'}, **m})
    
    pd.DataFrame(results).to_csv("results/lstm_benchmark.csv", index=False)