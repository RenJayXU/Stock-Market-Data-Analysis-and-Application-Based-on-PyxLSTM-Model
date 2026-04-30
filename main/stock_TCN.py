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
try:
    from metrics import calculate_metrics
except ImportError:
    print("請確認 metrics.py 是否存在於 main 資料夾中")
    exit()

# ==========================================
# 🔴 對齊 xLSTM 的超參數設定 
# ==========================================
CONFIG = {
    "stock_list": [
    # 前 10 大權重股
    '2330', '2317', '2454', '2308', '2382', '3711', '2303', '2891', '2881', '2882',
    
    # 金融控股族群
    '2886', '2884', '2892', '2890', '5880', '2883', '2887', '2885', '5871', '2880', '5876',
    
    # 電子、半導體與 AI 供應鏈 (含 Q3 新增: 2059)
    '2412', '3045', '4904', '3231', '2357', '2059', '2301', '2345', '3008', '2379', 
    '2408', '3034', '2327', '2474', '2395', '6669',
    
    # 航運與交通 (含 Q1 新增: 2615)
    '2603', '2609', '2615', '2207',
    
    # 生技醫療 (Q3 新增: 6919)
    '6919',
    
    # 塑化、鋼鐵、食品與傳產 (已剔除: 1326, 1101, 1760)
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'], # 建議先測 2330
    "input_size": 91,
    "num_channels": [128, 128], 
    "kernel_size": 3,
    "dropout": 0.2,            
    "sequence_length": 120,     
    "batch_size": 64,
    "learning_rate": 0.0001,   
    "num_epochs": 400,
    "patience": 50             
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
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

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
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # 🔴 將輸出維度設為 1，直接預測報酬率
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return self.fc(y[:, :, -1])

# --- 執行邏輯 ---
def run_tcn(stock_id, device):
    print(f"\n=== 執行 TCN Baseline 模型: {stock_id} ===")
    
    try:
        train_dataset = StockDataset(f"processed_data/train_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
        val_dataset = StockDataset(f"processed_data/val_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
        test_dataset = StockDataset(f"processed_data/test_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    except Exception as e:
        print(f"資料讀取錯誤: {e}")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TCNModel(CONFIG['input_size'], CONFIG['input_size'], CONFIG['num_channels'], CONFIG['kernel_size'], CONFIG['dropout']).to(device)
    
    # 🔴 改回純粹的 MSELoss
    train_loss_fn = nn.MSELoss()
    mse_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    wait = 0
    save_path = f"models/{stock_id}_tcn.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 🔴 維度防護
            if inputs.shape[-1] > 91:
                inputs = inputs[:, :, :-1]
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = train_loss_fn(outputs.squeeze(), targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if inputs.shape[-1] > 91:
                    inputs = inputs[:, :, :-1]
                    
                outputs = model(inputs)
                loss = mse_loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
             print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Val MSE: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("開始執行測試集預測...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    scaler = joblib.load(f"processed_data/{stock_id}_scaler.save")
    
    pred_return, actual_return, prev_actuals = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            if inputs.shape[-1] > 91:
                inputs = inputs[:, :, :-1]
                
            pred = model(inputs)
            pred_return.append(pred.squeeze().cpu().numpy())
            actual_return.append(targets[0].numpy())
            prev_actuals.append(inputs[0, -1, :].cpu().numpy())

    pred_return = np.array(pred_return)
    actual_return = np.array(actual_return)

    prev_inverse = scaler.inverse_transform(np.array(prev_actuals))
    prev_return = prev_inverse[:, 3]

    metrics = calculate_metrics(actual_return, pred_return, prev_return)
    print(f"Result {stock_id}: {metrics}")
    
    # 🔴 確保儲存格式一致
    df_test_raw = pd.read_csv(f"processed_data/test_{stock_id}.csv", index_col=0)
    test_dates = pd.to_datetime(df_test_raw.index)
    plot_dates = test_dates[-len(pred_return):]
    
    result_df = pd.DataFrame({
        "Date": plot_dates,
        "Actual_Return": actual_return,
        "Predicted_Return": pred_return
    })
    result_df.to_csv(f"results/tcn_prediction_{stock_id}.csv", index=False)
    
    return metrics

if __name__ == "__main__":
    device = torch.device("cpu") # 保持你原本設定的 CPU
    results = []
    for stock in CONFIG['stock_list']:
        m = run_tcn(stock, device)
        if m is not None:
            results.append({**{'Stock': stock, 'Model': 'TCN'}, **m})
    
    pd.DataFrame(results).to_csv("results/tcn_benchmark.csv", index=False)