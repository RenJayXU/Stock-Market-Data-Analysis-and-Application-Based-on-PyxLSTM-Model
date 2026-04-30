import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

STOCK_ID = "2330"

class StockDataset(Dataset):
    def __init__(self, csv_file, sequence_length=120, prediction_days=1):
        self.data = pd.read_csv(csv_file, index_col=0)
        
        # ==========================================
        # 🔴 核心修復：分離「特徵(X)」與「答案(y)」
        # ==========================================
        if 'Actual_Return' in self.data.columns:
            # 題目 (X)：剔除答案欄位，保證留下乾淨的 91 維
            self.X = self.data.drop(columns=['Actual_Return']).values.astype(np.float32)
            # 答案 (y)：單獨把未來報酬率抽出來
            self.y = self.data['Actual_Return'].values.astype(np.float32)
        else:
            # 防呆機制：萬一讀到還沒更新的舊 CSV
            self.X = self.data.values.astype(np.float32)
            self.y = np.zeros(len(self.data), dtype=np.float32)
        
        # 🔴 防護一：清除 NaN，對特徵 X 進行防護與裁切
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = np.clip(self.X, -5.0, 5.0) 
        
        # 答案 y 也做基本的防護
        self.targets = np.nan_to_num(self.y, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        
        # 計算可用索引
        self.valid_indices = len(self.features) - sequence_length - prediction_days + 1
        
        # 防護網
        if self.valid_indices < 0:
            self.valid_indices = 0

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # 輸入特徵序列：[120天, 91維度]
        input_seq = self.features[idx : idx + self.sequence_length]
        
        # 目標答案：直接抓取對應那一天的 Actual_Return (單一純量數值)
        target_idx = idx + self.sequence_length + self.prediction_days - 1
        target = self.targets[target_idx]
        
        return torch.tensor(input_seq), torch.tensor(target)

# (測試程式碼)
if __name__ == "__main__":
    try:
        train_dataset = StockDataset(f"processed_data/train_{STOCK_ID}.csv", sequence_length=120)
        print(f"訓練集總樣本數: {len(train_dataset)}")
        
        # 順便印出一個 Batch 檢查形狀
        x, y = train_dataset[0]
        print(f"X 特徵形狀: {x.shape} (應該要是 [120, 91])")
        print(f"y 答案形狀: {y.shape} (應該要是單一數值)")
    except Exception as e:
        print(f"\n[錯誤] {e}")