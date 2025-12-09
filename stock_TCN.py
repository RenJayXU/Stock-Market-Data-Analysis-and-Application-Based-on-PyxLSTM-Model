import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_dataset import StockDataset
from metrics import calculate_metrics

CONFIG = {
    "stock_list": ['1301', '2330', '2882', '1734', '3008'],
    "input_size": 9,
    "num_channels": [128, 128, 128], # 3層卷積，通道數對齊 hidden_size
    "kernel_size": 3,
    "dropout": 0.3,
    "sequence_length": 30,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_epochs": 200,
    "patience": 20
}

# --- TCN 模組定義 ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一次卷積
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二次卷積
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # TCN 預期輸入 (Batch, Channel, Length)
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (Batch, SeqLen, Features) -> 轉置為 (Batch, Features, SeqLen)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # 取最後一個時間點: y[:, :, -1]
        return self.fc(y[:, :, -1])

# --- 執行邏輯 ---
def run_tcn(stock_id, device):
    print(f"\n=== 執行 TCN 模型: {stock_id} ===")
    train_dataset = StockDataset(f"processed_data/train_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    val_dataset = StockDataset(f"processed_data/val_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    test_dataset = StockDataset(f"processed_data/test_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TCNModel(CONFIG['input_size'], 9, CONFIG['num_channels'], CONFIG['kernel_size'], CONFIG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    wait = 0
    save_path = f"models/{stock_id}_tcn.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs[:, 3], targets[:, 3]).item()
        
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

    # 評估
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    scaler = joblib.load(f"processed_data/{stock_id}_scaler.save")
    
    predictions, actuals, prev_actuals = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            pred = model(inputs)
            predictions.append(pred[0].cpu().numpy())
            actuals.append(targets[0].numpy())
            prev_actuals.append(inputs[0, -1, :].cpu().numpy())

    pred_close = scaler.inverse_transform(np.array(predictions))[:, 3]
    actual_close = scaler.inverse_transform(np.array(actuals))[:, 3]
    prev_close = scaler.inverse_transform(np.array(prev_actuals))[:, 3]

    metrics = calculate_metrics(actual_close, pred_close, prev_close)
    print(f"Result {stock_id}: {metrics}")
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for stock in CONFIG['stock_list']:
        m = run_tcn(stock, device)
        results.append({**{'Stock': stock, 'Model': 'TCN'}, **m})
    
    pd.DataFrame(results).to_csv("results/tcn_benchmark.csv", index=False)