import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os # 僅用於檢查檔案是否存在

# --- 【關鍵修正：導入同資料夾的 stock_predict】 ---
# 因為 stock_predict.py 在同一個資料夾，我們可以直接導入
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


# 數據集類 (與PyxLSTM共用，讀取已處理的CSV)
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

# LSTM模型定義 (9特徵)
class StockLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, output_size=9, dropout=0.3):
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
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        out = self.dropout_layer(out[:, -1, :]) 
        out = self.fc(out)
        return out, hidden

# 統一的訓練與評估函數
def train_and_evaluate_lstm(
    stock_id="1301",
    seq_length=30,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    batch_size=64,
    epochs=200,
    lr=0.0001,
    weight_decay=1e-5,
    patience=20
):
    
    # --- 1. 設置 ---
    print(f"=== 開始訓練 LSTM 模型 (對比 PyxLSTM) ===")
    print(f"Stock ID: {stock_id}, Features: 9")

    # 偵測 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 【關鍵修正：使用當前資料夾路徑】 ---
    # 假設所有檔案都在同一個資料夾
    train_csv_path = f'train_{stock_id}.csv'
    test_csv_path = f'test_{stock_id}.csv'
    scaler_path = f'{stock_id}_scaler.save'
    
    # 儲存最佳模型的路徑 (存在當前資料夾)
    MODEL_SAVE_PATH = 'best_lstm_model.pth'
    # --- 【修正結束】 ---

    # --- 2. 載入資料 ---
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {scaler_path}。")
        print(f"請確認您已在 '此資料夾' 中執行 'stock_preprocessing.py'，")
        print(f"且已產生 '{stock_id}_scaler.save' 檔案。")
        return
    
    try:
        train_dataset = StockDataset(train_csv_path, seq_length)
        val_dataset = StockDataset(test_csv_path, seq_length) # 使用 test_set 作為 validation
    except FileNotFoundError:
        print(f"錯誤: 找不到 {train_csv_path} 或 {test_csv_path}。")
        print(f"請確認 'train_{stock_id}.csv' 和 'test_{stock_id}.csv' 也在這個資料夾。")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"訓練集: {len(train_dataset)} 樣本, 驗證集: {len(val_dataset)} 樣本")

    # --- 3. 初始化模型 ---
    model = StockLSTM(
        input_size=9,
        hidden_size=hidden_size,
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
    
    print("\n=== 開始訓練 LSTM ===")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
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
                outputs, _ = model(inputs)
                # 只計算 'Close' (索引 3) 的 Loss 來決定最佳模型
                loss = criterion_val(outputs[:, 3], targets[:, 3])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss (Close): {avg_val_loss:.6f}')
        
        # 早停邏輯
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'*** New best model (LSTM) saved at Epoch {epoch+1} ***')
        else:
            wait += 1
            if wait >= patience:
                print(f'Val Loss 連續 {patience} 個 epoch 未改善，提前停止。')
                break
    
    end_time = time.time()
    print(f"\n訓練完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")

    # --- 5. 最終評估 (載入最佳模型) ---
    print(f"\n=== 最終評估 (載入最佳 LSTM 模型) ===")
    
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
            outputs, _ = model(inputs)
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
    
    print(f'\n=== LSTM 最終測試結果 (Stock: {stock_id}) ===')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {acc:.4f}')
    
    # 可視化
    plt.figure(figsize=(18, 9))
    
    plt.subplot(2, 1, 1)
    plt.plot(actual_close, label='Actual Closing Price', linewidth=2)
    plt.plot(pred_close, label='Predicted Closing Price (LSTM)', linewidth=2, linestyle='--')
    plt.title(f'{stock_id} Price Forecast Comparison (LSTM)', fontsize=16)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(actual_labels, 'g-', label='Actual Trend')
    plt.plot(pred_labels, 'r--', label='Predicted Trend (LSTM)')
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.title('Trend Prediction Accuracy (LSTM)', fontsize=14)
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Trend', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{stock_id}lstm_prediction_results.png')
    plt.show()

if __name__ == '__main__':
    # 使用與 PyxLSTM 相同的參數配置
    # 確保這些參數與您 PyxLSTM 訓練時的 config 一致
    train_and_evaluate_lstm(
        stock_id="3008",
        seq_length=30,
        hidden_size=128, 
        num_layers=2,    
        dropout=0.3, # 保持與 PyxLSTM 的 config 一致 (0.3 或 0.4)
        batch_size=64,
        epochs=200,
        lr=0.0001,
        weight_decay=1e-5,
        patience=20
    )