import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from stock_xlstm import StockxLSTM
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

STOCK_LIST = [
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
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'
]

# ==========================================
# 終極收益率損失函數
# ==========================================
class WeightedReturnLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=50.0, gamma=10.0, up_weight=1.5, margin=0.1):
        super(WeightedReturnLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha      
        self.beta = beta        
        self.gamma = gamma      
        self.up_weight = up_weight
        self.margin = margin    

    def forward(self, outputs, targets):
        # 🔴 核心修正 1：指定輸出的第 0 個節點作為預測值，並直接使用乾淨的 targets
        y_pred = outputs[:, 0]
        y_true = targets
        
        noise = 1e-7 * torch.randn_like(y_pred)
        y_pred = y_pred + noise
        
        # 1. 基礎 MSE
        base_loss = self.mse(y_pred, y_true) * 0.1 
        
        # 2. 方向懲罰
        direction_penalty = torch.relu(self.margin - torch.sign(y_true) * y_pred)
        is_up_day = (y_true > 0).float()
        weights = is_up_day * self.up_weight + (1 - is_up_day) * 1.0
        direction_loss = torch.mean(weights * direction_penalty)
        
        # 3. 波動度懲罰
        std_pred = torch.std(y_pred) + 1e-6
        std_true = torch.std(y_true) + 1e-6
        variance_loss = torch.pow(std_true - std_pred, 2) 
        
        # 4. 形狀相關性損失
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        corr = F.cosine_similarity(vx.unsqueeze(0), vy.unsqueeze(0), eps=1e-8)
        corr_loss = 1.0 - corr.squeeze() 
        
        if torch.isnan(corr_loss):
            corr_loss = torch.tensor(2.0, device=y_pred.device, requires_grad=True)
            
        return base_loss + (self.alpha * direction_loss) + (self.beta * variance_loss) + (self.gamma * corr_loss)

def train_stock_model(stock_id, device):
    print(f"\n{'='*20} 開始訓練股票: {stock_id} {'='*20}")
    
    config = {
        "sequence_length": 120,    
        "prediction_days": 1,
        "batch_size": 64,          
        "hidden_size": 128,        
        "num_layers": 1,
        "num_blocks": 2,
        "dropout": 0.1,            
        "learning_rate": 0.0001,   
        "weight_decay": 1e-4,      
        "num_epochs": 400,
        "patience": 50,            
        "lstm_type": ['mlstm','slstm'] 
    }

    try:
        train_dataset = StockDataset(
            f"processed_data/train_{stock_id}.csv",
            sequence_length=config["sequence_length"],
            prediction_days=config["prediction_days"],
        )
        val_dataset = StockDataset(
            f"processed_data/val_{stock_id}.csv",
            sequence_length=config["sequence_length"],
            prediction_days=config["prediction_days"],
        )

        if len(train_dataset) <= 0 or len(val_dataset) <= 0:
            print(f"跳過 {stock_id}: 資料量不足")
            return
            
    except FileNotFoundError:
        print(f"找不到 {stock_id} 的處理後資料")
        return
    except Exception as e:
        print(f"錯誤: {e}")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model = StockxLSTM(
        input_size=91,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
        lstm_type=config["lstm_type"] 
    ).to(device)

# 🔴 復健期：移除所有花俏的懲罰，讓模型專心學習基礎的 MSE
    train_loss_fn = nn.MSELoss() 
    mse_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_loss_history, val_loss_history = [], []
    best_val_loss = float("inf")
    wait = 0
    
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{stock_id}_best_model.pth"

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_train_loss_target = 0.0 
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 🔴 確保模型只拿第 0 個預測節點來算純粹的 MSE
            loss = train_loss_fn(outputs[:, 0], targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 🔴 核心修正 2：計算純 MSE 時，一樣使用 outputs[:, 0] 與 targets
            loss_target = mse_loss_fn(outputs[:, 0], targets)
            epoch_train_loss_target += loss_target.item()
        
        avg_train_loss_target = epoch_train_loss_target / len(train_loader)
        train_loss_history.append(avg_train_loss_target)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # 🔴 核心修正 3：驗證集也使用 outputs[:, 0] 與 targets
                loss = mse_loss_fn(outputs[:, 0], targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{config['num_epochs']} | LR: {current_lr:.6f} | "
                  f"Train MSE: {avg_train_loss_target:.6f} | Val MSE: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train MSE")
    plt.plot(val_loss_history, label="Val MSE")
    plt.title(f"{stock_id} (Seq:120) Loss History")
    plt.legend()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{stock_id}_loss.png")
    plt.close()
    
    print(f"訓練完成: {stock_id}, 最佳 Val Loss: {best_val_loss:.6f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for stock in tqdm(STOCK_LIST, desc="總股票訓練進度"):
        train_stock_model(stock, device)

if __name__ == "__main__":
    main()