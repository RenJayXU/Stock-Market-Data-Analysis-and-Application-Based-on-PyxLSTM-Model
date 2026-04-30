import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
import os
import sys
import math

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
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'],
    "input_size": 91,
    "hidden_size": 128,
    "num_heads": 4,
    "num_layers": 1,           
    "dropout": 0.2,            
    "sequence_length": 120,     
    "batch_size": 64,
    "learning_rate": 0.0001,   
    "num_epochs": 400,
    "patience": 50             
}

# --- Transformer 模組定義 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # 🔴 將輸出維度設為 1，直接預測報酬率
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        src = src.transpose(0, 1) 
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[-1])

# --- 執行邏輯 ---
def run_transformer(stock_id, device):
    print(f"\n=== 執行 Transformer Baseline 模型: {stock_id} ===")
    
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

    model = TransformerModel(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_heads'], 
                             CONFIG['num_layers'], CONFIG['dropout']).to(device)
    
    # 🔴 改回純粹的 MSELoss
    train_loss_fn = nn.MSELoss()
    mse_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    wait = 0
    save_path = f"models/{stock_id}_transformer.pth"
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
    result_df.to_csv(f"results/transformer_prediction_{stock_id}.csv", index=False)
    
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for stock in CONFIG['stock_list']:
        m = run_transformer(stock, device)
        if m is not None:
            results.append({**{'Stock': stock, 'Model': 'Transformer'}, **m})
    
    pd.DataFrame(results).to_csv("results/transformer_benchmark.csv", index=False)