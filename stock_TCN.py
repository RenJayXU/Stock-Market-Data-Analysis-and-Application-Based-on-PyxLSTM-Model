import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ====== 數據預處理（避免Data Leakage） ======
class StockPreprocessor:
    def __init__(self, train_ratio=0.7, val_ratio=0.2):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def process(self, data_path):
        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        alldates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(alldates)
        df = df.ffill()
        train_size = int(len(df) * self.train_ratio)
        val_size = int(len(df) * self.val_ratio)
        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size + val_size]
        test_data = df.iloc[train_size + val_size:]
        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)
        test_scaled = self.scaler.transform(test_data)
        return train_scaled, val_scaled, test_scaled

class StockDataset(Dataset):
    def __init__(self, data, window_size=30):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]
        target = self.data[idx+self.window_size]
        return torch.FloatTensor(window), torch.FloatTensor(target)

from torch.nn.utils.parametrizations import weight_norm

class Crop(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size
    def forward(self, x):
        return x[:, :, :-self.crop_size] if self.crop_size != 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.crop1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.crop2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size=5, output_size=5, num_channels=[64,64], kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        out = self.network(x)
        out = self.linear(out[:, :, -1])
        return out

def train_tcn(
    data_csv='2882.csv',
    model_save_path='2882tcn_fixed_model.pth',
    seq_length=30,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    patience=15
):
    preprocessor = StockPreprocessor()
    train_data, val_data, test_data = preprocessor.process(data_csv)
    print(f"Training: {len(train_data)} rows | Val: {len(val_data)} rows | Test: {len(test_data)} rows")
    train_dataset = StockDataset(train_data, seq_length)
    val_dataset = StockDataset(val_data, seq_length)
    test_dataset = StockDataset(test_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TCN(input_size=5, output_size=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
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
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | ★ Best')
        else:
            patience_counter += 1
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Patience: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print("\n=== 最終測試評估 ===")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    predictions, actuals = [], []
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
    pred_close = preprocessor.scaler.inverse_transform(dummy_pred)[:, 3]
    dummy_actual = np.zeros((len(actuals), 5))
    dummy_actual[:, 3] = actuals[:, 3]
    actual_close = preprocessor.scaler.inverse_transform(dummy_actual)[:, 3]

    from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score
    mse = mean_squared_error(actual_close, pred_close)
    mae = mean_absolute_error(actual_close, pred_close)
    r2 = 1 - mse / np.var(actual_close)
    if len(actual_close) > 1:
        actual_labels = np.diff(actual_close) > 0
        pred_labels = np.diff(pred_close) > 0
        f1 = f1_score(actual_labels, pred_labels, zero_division=0)
        acc = accuracy_score(actual_labels, pred_labels)
    else:
        f1, acc = 0, 0
    print(f'\n=== 最終測試結果 ===')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {acc:.4f}')

    # 畫圖
    plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(actual_close, label='Actual', alpha=0.8)
    plt.plot(pred_close, label='Predicted', alpha=0.8)
    plt.title('Stock Price Prediction (Test Set Only)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    if len(actual_close) > 1:
        plt.subplot(3, 1, 3)
        plt.plot(actual_labels.astype(int), label='Actual Trend', alpha=0.8)
        plt.plot(pred_labels.astype(int), label='Predicted Trend', alpha=0.8)
        plt.title('Up/Down Trend Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Trend (1=Up, 0=Down)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    return pred_close, actual_close, train_losses, val_losses

if __name__ == '__main__':
    tcn_pred, tcn_actual, train_loss_history, val_loss_history = train_tcn(
        data_csv='2882.csv',
        model_save_path='2882tcn_fixed_model.pth',
        seq_length=30,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001,
        patience=15
    )
