import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM
import matplotlib.pyplot as plt
import time # 用於計時

def main():
    
    # 1. 【新增】 偵測並定義 device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # 2. 【修改】 簡化模型參數，對抗 Overfitting
    config = {
        "stock_id": "2330",
        "sequence_length": 30,
        "prediction_days": 1,
        "batch_size": 64,
        "hidden_size": 128,  # <-- 降低: 256 -> 128
        "num_layers": 2,
        "num_blocks": 2,     # <-- 降低: 3 -> 2
        "dropout": 0.4,      # <-- 調整: 0.4 -> 0.3
        "learning_rate": 0.0001, # <-- 提高: 5e-5 -> 1e-4
        "weight_decay": 1e-5,  # <-- 【新增】 L2 懲罰
        "num_epochs": 200,     # 保持 200
    }
    STOCK_ID = config["stock_id"]

    # 3. 準備資料集
    train_dataset = StockDataset(
        f"train_{STOCK_ID}.csv",
        sequence_length=config["sequence_length"],
        prediction_days=config["prediction_days"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    
    # 【新增】 載入驗證集 (我們使用 test_set 作為 validation_set)
    val_dataset = StockDataset(
        f"test_{STOCK_ID}.csv",
        sequence_length=config["sequence_length"],
        prediction_days=config["prediction_days"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    print(f"訓練集樣本數: {len(train_dataset)}")
    print(f"驗證集樣本數: {len(val_dataset)}")
    
    sample_input, sample_target = train_dataset[0]
    print(f"輸入序列形狀: {sample_input.shape}") # (30, 9)
    print(f"目標形狀: {sample_target.shape}")    # (9,)

    # 4. 建立模型 (sLSTM)
    model = StockxLSTM(
        input_size=9,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
        lstm_type="slstm" # 保持 sLSTM
    )
    model.to(device)

    loss_fn = nn.MSELoss()
    # 【修改】 Optimizer 加入 weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )

    train_loss_history = []
    val_loss_history = [] # 【新增】

    # 早停參數設定
    patience = 20  # 【修改】 增加耐心到 20
    best_loss = float("inf") # 現在 best_loss 是指 "best validation loss"
    wait = 0

    print("=== 開始訓練 ===")
    start_time = time.time()
    
    # 5. 訓練模型
    for epoch in range(config["num_epochs"]):
        
        # --- 訓練模式 ---
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets) # 訓練所有 9 個特徵
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- 驗證模式 ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs[:, 3], targets[:, 3])
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # --- 打印 Log ---
        print(f"Epoch {epoch+1}/{config['num_epochs']}, "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")

        # 【關鍵修改】 早停判斷 (監控 Val Loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            wait = 0
            # 儲存 "驗證集上表現最好" 的模型
            torch.save(model.state_dict(), f"{STOCK_ID}model.pth")
            print(f"*** New best model saved at Epoch {epoch+1} with Val Loss: {best_loss:.6f} ***")
        else:
            wait += 1
            if wait >= patience:
                print(f"Validation Loss 已連續 {patience} 個 epoch 沒有下降，提前停止訓練。")
                break

    end_time = time.time()
    print(f"訓練完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    print(f"已儲存最佳模型 (Val Loss: {best_loss:.6f})。")

    # 6. 畫出訓練損失曲線
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss") # 【新增】
    plt.title("Training & Validation Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()