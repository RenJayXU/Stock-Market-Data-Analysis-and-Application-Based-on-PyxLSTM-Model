import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM
import matplotlib.pyplot as plt


def main():
    config = {
        "sequence_length": 30,
        "prediction_days": 1,
        "batch_size": 64,
        "hidden_size": 64,
        "num_layers": 2,  # 2
        "num_blocks": 3,
        "dropout": 0.3,  # 0.3
        "learning_rate": 0.0001,  # 0.00003
        "num_epochs": 100,  # 100
    }

    # 1. 準備資料集
    train_dataset = StockDataset(
        "train_2330.csv",
        sequence_length=config["sequence_length"],
        prediction_days=config["prediction_days"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    print(f"訓練集樣本數: {len(train_dataset)}")
    sample_input, sample_target = train_dataset[0]
    print(f"輸入序列形狀: {sample_input.shape}")
    print(f"目標形狀: {sample_target.shape}")

    # 2. 建立模型
    model = StockxLSTM(
        input_size=5,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )

    loss_fn = nn.MSELoss()  # 多特徵均方誤差
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loss_history = []

    # 早停參數設定
    patience = 10  # 連續多少個epoch loss不下降就停止
    best_loss = float("inf")
    wait = 0

    # 3. 訓練模型
    model.train()
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            # 這裡直接對close特徵做優化
            loss = loss_fn(outputs[:, 3], targets[:, 3])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}")

        # 早停判斷
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"Loss 已連續 {patience} 個 epoch 沒有下降，提前停止訓練。"
                )
                break

    # 4. 儲存模型
    torch.save(model.state_dict(), "2330model.pth")

    # 5. 畫出訓練損失曲線
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label="Training Loss")
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
