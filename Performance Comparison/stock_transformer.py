import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os

# --- 【導入同資料夾的 stock_predict】 ---
try:
    from stock_predict import safe_r2_score, calculate_trend_metrics
except ImportError:
    print("錯誤：無法從 'stock_predict.py' 導入函數。")
    print("請確保 'stock_predict.py' 與此腳本在同一個資料夾中。")
    # 如果導入失敗，手動定義 (備用方案)
    def safe_r2_score(y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10: return float('nan')
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_trend_metrics(actuals, preds):
        from sklearn.metrics import f1_score, accuracy_score
        actual_labels = (np.diff(actuals) > 0).astype(int)
        pred_labels = (np.diff(preds) > 0).astype(int)
        min_length = min(len(actual_labels), len(pred_labels))
        actual_labels = actual_labels[:min_length]
        pred_labels = pred_labels[:min_length]
        f1 = f1_score(actual_labels, pred_labels, zero_division=0)
        acc = accuracy_score(actual_labels, pred_labels)
        return f1, acc, actual_labels, pred_labels
# --- 【修正結束】 ---

# --- 【新增 StockDataset 類 (使用9特徵)】 ---
class StockDataset(Dataset):
    def __init__(self, data_path, sequence_length=30):
        # 讀取已標準化和包含9特徵的CSV
        self.data = pd.read_csv(data_path, index_col=0).values.astype(np.float32)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        window = self.data[idx : idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return torch.FloatTensor(window), torch.FloatTensor(target)
# --- 【修正結束】 ---


# 位置編碼層 (Transformer關鍵組件)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Transformer模型定義 (修改為 9 特徵)
class StockTransformer(nn.Module):
    def __init__(self, input_size=9, d_model=128, nhead=8, num_layers=2, output_size=9, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model) # 嵌入層
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder (必須指定 batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 輸出層
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size) # 線性輸出
        )

    def forward(self, src):
        # src shape: (batch_size, seq_length, input_size)
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # output shape: (batch_size, seq_length, d_model)
        output = self.transformer_encoder(src) 
        
        # 【修改】只取最後一個時間步的輸出來預測 (比 .mean() 更好)
        output = output[:, -1, :]  
        return self.fc(output)

# 統一的訓練與評估函數
def train_and_evaluate_transformer(
    stock_id="2330",
    seq_length=30,
    d_model=128,      # 匹配 hidden_size
    nhead=8,
    num_layers=2,     # 匹配 num_layers
    dropout=0.3,
    batch_size=64,
    epochs=200,
    lr=0.0001,
    weight_decay=1e-5,
    patience=20
):
    
    # --- 1. 設置 ---
    print(f"=== 開始訓練 Transformer 模型 (對比 PyxLSTM) ===")
    print(f"Stock ID: {stock_id}, Features: 9")

    # 偵測 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 【關鍵修正：使用當前資料夾路徑】 ---
    train_csv_path = f'train_{stock_id}.csv'
    test_csv_path = f'test_{stock_id}.csv'
    scaler_path = f'{stock_id}_scaler.save'
    
    # 儲存最佳模型的路徑
    MODEL_SAVE_PATH = 'best_transformer_model.pth'
    # --- 【修正結束】 ---

    # --- 2. 載入資料 ---
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {scaler_path}。")
        print(f"請確認您已在 '此資料夾' 中執行 'stock_preprocessing.py'。")
        return
    
    try:
        train_dataset = StockDataset(train_csv_path, seq_length)
        val_dataset = StockDataset(test_csv_path, seq_length)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {train_csv_path} 或 {test_csv_path}。")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"訓練集: {len(train_dataset)} 樣本, 驗證集: {len(val_dataset)} 樣本")
    
    # --- 3. 初始化模型 ---
    model = StockTransformer(
        input_size=9,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_size=9,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() # 總體 MSE
    criterion_val = nn.MSELoss() # 驗證 Close MSE
    
    # --- 4. 訓練迴圈 (含驗證集早停) ---
    best_val_loss = float('inf')
    wait = 0
    start_time = time.time()
    
    print("\n=== 開始訓練 Transformer ===")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) # 訓練所有 9 個特徵
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # 【修改】只計算 'Close' (索引 3) 的 Loss 來決定最佳模型
                loss = criterion_val(outputs[:, 3], targets[:, 3])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss (Close): {avg_val_loss:.6f}')
        
        # 早停邏輯
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'*** New best model (Transformer) saved at Epoch {epoch+1} ***')
        else:
            wait += 1
            if wait >= patience:
                print(f'Val Loss 連續 {patience} 個 epoch 未改善，提前停止。')
                break
                
    end_time = time.time()
    print(f"\n訓練完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    
    # --- 5. 最終評估 (載入最佳模型) ---
    print(f"\n=== 最終評估 (載入最佳 Transformer 模型) ===")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"錯誤：找不到儲存的模型 {MODEL_SAVE_PATH}。可能是訓練一開始就失敗了。")
        return
        
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # 反標準化 (只看 'Close'，索引 3)
    dummy_pred = np.zeros_like(predictions)
    dummy_pred[:, :] = predictions
    pred_close = scaler.inverse_transform(dummy_pred)[:, 3]
    
    dummy_actual = np.zeros_like(actuals)
    dummy_actual[:, :] = actuals
    actual_close = scaler.inverse_transform(dummy_actual)[:, 3]
    
    # 計算指標
    mse = mean_squared_error(actual_close, pred_close)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = safe_r2_score(actual_close, pred_close)
    f1, acc, actual_labels, pred_labels = calculate_trend_metrics(actual_close, pred_close)
    
    print(f'\n=== Transformer 最終測試結果 (Stock: {stock_id}) ===')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {acc:.4f}')
    
    # 可視化
    plt.figure(figsize=(18, 9))
    
    plt.subplot(2, 1, 1)
    plt.plot(actual_close, label='Actual Closing Price', linewidth=2)
    plt.plot(pred_close, label='Predicted Closing Price (Transformer)', linewidth=2, linestyle='--')
    plt.title(f'{stock_id} Price Forecast Comparison (Transformer)', fontsize=16)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(actual_labels, 'g-', label='Actual Trend')
    plt.plot(pred_labels, 'r--', label='Predicted Trend (Transformer)')
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.title('Trend Prediction Accuracy (Transformer)', fontsize=14)
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Trend', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{stock_id}transformer_prediction_results.png')
    plt.show()

if __name__ == '__main__':
    # 使用與 PyxLSTM 相同的參數配置
    train_and_evaluate_transformer(
        stock_id="3008",
        seq_length=30,
        d_model=128,     # 匹配 hidden_size
        nhead=8,
        num_layers=2,    # 匹配 num_layers
        dropout=0.3,     # 匹配 dropout
        batch_size=64,
        epochs=200,
        lr=0.0001,
        weight_decay=1e-5,
        patience=20
    )