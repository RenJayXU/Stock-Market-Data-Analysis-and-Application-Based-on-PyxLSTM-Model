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
from metrics import calculate_metrics

CONFIG = {
    "stock_list": ['1301', '2330', '2882', '1734', '3008'],
    "input_size": 9,
    "hidden_size": 128,  # d_model
    "num_heads": 4,      # Attention heads
    "num_layers": 2,     # Encoder layers
    "dropout": 0.3,
    "sequence_length": 30,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_epochs": 200,
    "patience": 20
}

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
        # x: (SeqLen, Batch, Feature)
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 9)
        self.d_model = d_model

    def forward(self, src):
        # src: (Batch, SeqLen, InputSize)
        # Transformer 預設輸入為 (SeqLen, Batch, Feature)
        src = src.transpose(0, 1) 
        
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # 取最後一個時間點的輸出 output[-1] -> (Batch, d_model)
        return self.decoder(output[-1])

def run_transformer(stock_id, device):
    print(f"\n=== 執行 Transformer 模型: {stock_id} ===")
    train_dataset = StockDataset(f"processed_data/train_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    val_dataset = StockDataset(f"processed_data/val_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    test_dataset = StockDataset(f"processed_data/test_{stock_id}.csv", sequence_length=CONFIG['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TransformerModel(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_heads'], 
                             CONFIG['num_layers'], CONFIG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    wait = 0
    save_path = f"models/{stock_id}_transformer.pth"
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
        m = run_transformer(stock, device)
        results.append({**{'Stock': stock, 'Model': 'Transformer'}, **m})
    
    pd.DataFrame(results).to_csv("results/transformer_benchmark.csv", index=False)