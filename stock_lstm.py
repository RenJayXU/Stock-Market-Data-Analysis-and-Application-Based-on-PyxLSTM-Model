import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from stock_predict import safe_r2_score, calculate_trend_metrics

# 數據集類(與PyxLSTM共用)
class StockDataset(Dataset):
    def __init__(self, data_path, sequence_length=30):
        self.data = pd.read_csv(data_path, index_col=0).values
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        window = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+self.sequence_length]
        return torch.FloatTensor(window), torch.FloatTensor(target)

# LSTM模型定義
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, dropout=0.3):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])  # 取最後時間步
        out = self.fc(out)
        return out, hidden

def train_lstm(
    train_csv_path='train_3008.csv',
    test_csv_path='test_3008.csv',
    scaler_path='3008_scaler.save',
    seq_length=30,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    batch_size=64,
    epochs=150,
    lr=0.0005
):
    # 加載標準化器
    scaler = joblib.load(scaler_path)
    
    # 數據集與數據加載器
    train_dataset = StockDataset(train_csv_path, seq_length)
    test_dataset = StockDataset(test_csv_path, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型初始化
    model = StockLSTM(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 訓練循環
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 每個epoch驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs, _ = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(test_loader):.6f}')
    
    # 最終測試
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, _ = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # 反標準化
    dummy_pred = np.zeros((len(predictions), 5))
    dummy_pred[:, 3] = predictions[:, 3]  # 假設Close在第4列
    pred_close = scaler.inverse_transform(dummy_pred)[:, 3]
    
    dummy_actual = np.zeros((len(actuals), 5))
    dummy_actual[:, 3] = actuals[:, 3]
    actual_close = scaler.inverse_transform(dummy_actual)[:, 3]
    
    # 計算指標
    mse = mean_squared_error(actual_close, pred_close)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = safe_r2_score(actual_close, pred_close)
    f1, acc, actual_labels, pred_labels = calculate_trend_metrics(actual_close, pred_close)
    
    print(f'\nFinal Test Results:')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {acc:.4f}')
    
    # 可視化
    plt.figure(figsize=(18, 9))
    
    plt.subplot(2, 1, 1)
    plt.plot(actual_close, label='Actual Closing Price', linewidth=2, marker='o', markersize=5)
    plt.plot(pred_close, label='Predicted Closing Price', linewidth=2, linestyle='--', marker='x', markersize=5)
    plt.title('3008 Price Forecast Comparison (LSTM)', fontsize=16)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
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
    plt.savefig('lstm_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_close, actual_close

if __name__ == '__main__':
    lstm_pred, lstm_actual = train_lstm()
