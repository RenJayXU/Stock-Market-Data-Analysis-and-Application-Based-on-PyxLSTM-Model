import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from stock_dataset import StockDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from stock_predict import safe_r2_score, calculate_trend_metrics


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


# Transformer模型定義
class StockTransformer(nn.Module):
    def __init__(self, input_size=5, d_model=64, nhead=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)
        )

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # 全局平均池化
        return self.fc(output)


def train_transformer(
    train_csv_path='train_2882.csv',
    test_csv_path='test_2882.csv',
    scaler_path='2882_scaler.save',
    seq_length=30,
    d_model=64,
    nhead=8,
    num_layers=4,
    dropout=0.3,
    batch_size=64,
    epochs=100,
    lr=0.0005,
    patience=10,  # Early stopping patience
    min_delta=1e-6  # 最小改善量
):
    # 加載標準化器
    scaler = joblib.load(scaler_path)
    
    # 數據集與數據加載器
    train_dataset = StockDataset(train_csv_path, seq_length)
    test_dataset = StockDataset(test_csv_path, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型初始化
    model = StockTransformer(
        input_size=5,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Early Stopping 變數
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # 訓練循環
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}')
        
        # Early Stopping 邏輯
        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_transformer.pth')
            print(f'*** New best model saved at epoch {best_epoch} with val_loss: {best_loss:.6f} ***')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (best: {best_loss:.6f} at epoch {best_epoch})')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered! No improvement for {patience} epochs.')
                print(f'Best model was at epoch {best_epoch} with validation loss: {best_loss:.6f}')
                break
    
    # 最終測試
    print(f'\nLoading best model from epoch {best_epoch}...')
    model.load_state_dict(torch.load('best_transformer.pth'))
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # 反標準化
    dummy_pred = np.zeros((len(predictions), 5))
    dummy_pred[:, 3] = predictions[:, 3]
    pred_close = scaler.inverse_transform(dummy_pred)[:, 3]
    
    dummy_actual = np.zeros((len(actuals), 5))
    dummy_actual[:, 3] = actuals[:, 3]
    actual_close = scaler.inverse_transform(dummy_actual)[:, 3]
    
    # 計算指標
    mse = mean_squared_error(actual_close, pred_close)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = safe_r2_score(actual_close, pred_close)
    f1, acc, actual_labels, pred_labels = calculate_trend_metrics(actual_close, pred_close)
    
    print(f'\nTransformer Test Results (Best Model from Epoch {best_epoch}):')
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
    plt.title(f'2882 Price Forecast Comparison (Transformer - Best Epoch {best_epoch})', fontsize=16)
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
    plt.savefig('transformer_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_close, actual_close


if __name__ == '__main__':
    transformer_pred, transformer_actual = train_transformer()
