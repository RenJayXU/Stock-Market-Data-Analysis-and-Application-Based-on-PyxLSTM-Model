import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, csv_file, sequence_length=30, prediction_days=1):
        self.data = pd.read_csv(csv_file, index_col=0)
        self.features = self.data[["Open","High","Low","Close","Volume"]].values.astype(np.float32)
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.valid_indices = len(self.features) - sequence_length - prediction_days + 1

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        input_seq = self.features[idx:idx+self.sequence_length]
        target_idx = idx + self.sequence_length + self.prediction_days - 1
        target = self.features[target_idx]  # 取完整5維目標
        return torch.tensor(input_seq), torch.tensor(target)
    
    # 新增方法：獲取完整特徵的目標值
    def get_full_target(self, idx):
        target_idx = idx + self.sequence_length + (self.prediction_days - 1)
        return torch.tensor(self.features[target_idx])

# 建立訓練集與測試集的Dataset
train_dataset = StockDataset("train_2330.csv", sequence_length=30)
test_dataset = StockDataset("test_2330.csv", sequence_length=30)

# （可選）檢查資料集大小與樣本格式
print(f"訓練集樣本數: {len(train_dataset)}")
print(f"測試集樣本數: {len(test_dataset)}")
sample_input, sample_target = train_dataset[0]
print(f"輸入序列形狀: {sample_input.shape}")  # 應為 (30, 5)
print(f"目標形狀: {sample_target.shape}")    # 應為 (5,)


# 診斷步驟2：檢查StockDataset返回的實際值
test_dataset = StockDataset("test_2330.csv", sequence_length=30)
print("\n=== 目標值驗證 ===")
for i in range(3):  # 只顯示前3個樣本
    input_seq, target = test_dataset[i]
    print(f"樣本{i} 輸入形狀: {input_seq.shape}, 目標形狀: {target.shape}")
    print(f"目標值: {target.numpy().round(4)}")  # 直接顯示數組

