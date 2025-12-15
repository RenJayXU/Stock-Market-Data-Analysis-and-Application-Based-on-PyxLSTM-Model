import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM
import matplotlib.pyplot as plt
import time
import os

# 定義股票列表
STOCK_LIST = ['1301', '2330', '2882', '1734', '3008']

def train_stock_model(stock_id, device):
    print(f"\n{'='*20} 開始訓練股票: {stock_id} {'='*20}")
    
    # 參數設定
    config = {
        "sequence_length": 30,
        "prediction_days": 1,
        "batch_size": 64,
        "hidden_size": 128, 
        "num_layers": 1,
        "num_blocks": 2,
        "dropout": 0.3,      
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_epochs": 200,
        "patience": 20,
        # 新增混合模式設定
        "lstm_type": ['mlstm', 'slstm'] 
    }

    # 1. 準備資料集 (讀取 processed_data 資料夾)
    try:
        train_dataset = StockDataset(
            f"processed_data/train_{stock_id}.csv",
            sequence_length=config["sequence_length"],
            prediction_days=config["prediction_days"],
        )
        # 【修正】使用獨立的驗證集 val_xxx.csv
        val_dataset = StockDataset(
            f"processed_data/val_{stock_id}.csv",
            sequence_length=config["sequence_length"],
            prediction_days=config["prediction_days"],
        )
    except FileNotFoundError:
        print(f"找不到 {stock_id} 的處理後資料，請先執行 stock_preprocessing.py")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # 2. 建立模型
    model = StockxLSTM(
        input_size=9,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
        lstm_type=config["lstm_type"]  # <--- 這裡修改
    )
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )

    train_loss_history = []  # 記錄 Close Price 的 Loss
    val_loss_history = []

    best_val_loss = float("inf")
    wait = 0
    
    # 建立模型儲存資料夾
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{stock_id}_best_model.pth"

    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        # --- 訓練模式 ---
        model.train()
        epoch_train_loss_all = 0.0   # 所有特徵的 loss (用於反向傳播)
        epoch_train_loss_close = 0.0 # 僅 Close 的 loss (用於觀察)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_fn(outputs, targets) # 優化所有 9 個特徵
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_loss_all += loss.item()
            # 額外計算 Close Price (index 3) 的 Loss
            loss_close = loss_fn(outputs[:, 3], targets[:, 3])
            epoch_train_loss_close += loss_close.item()
        
        avg_train_loss_close = epoch_train_loss_close / len(train_loader)
        train_loss_history.append(avg_train_loss_close)

        # --- 驗證模式 ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # 驗證只看 Close Price 的準度
                loss = loss_fn(outputs[:, 3], targets[:, 3])
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Train Close Loss: {avg_train_loss_close:.6f} | "
                  f"Val Close Loss: {avg_val_loss:.6f}")

        # 早停機制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            # print(f"  -> Model saved (Val Loss: {best_val_loss:.6f})")
        else:
            wait += 1
            if wait >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 畫圖
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss (Close)")
    plt.plot(val_loss_history, label="Val Loss (Close)")
    plt.title(f"{stock_id} Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    # 儲存圖片到 results 資料夾
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{stock_id}_loss.png")
    plt.close()
    
    print(f"訓練完成: {stock_id}, 最佳 Val Loss: {best_val_loss:.6f}")

def main():
    # 偵測 Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for stock in STOCK_LIST:
        train_stock_model(stock, device)

if __name__ == "__main__":
    main()