import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

STOCK_ID = "1301"

class StockDataset(Dataset):
    def __init__(self, csv_file, sequence_length=30, prediction_days=1):
        self.data = pd.read_csv(csv_file, index_col=0)
        # 【修改】更新特徵列表
        self.features_list = [
            "Open", "High", "Low", "Close", "Volume",
            "RSI_14", "MACD_12_26_9", "SMA_10", "SMA_20"
        ]
        self.features = self.data[self.features_list].values.astype(np.float32)
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.valid_indices = len(self.features) - sequence_length - prediction_days + 1

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        input_seq = self.features[idx:idx+self.sequence_length]
        target_idx = idx + self.sequence_length + self.prediction_days - 1
        target = self.features[target_idx] # 取完整 9 維目標
        return torch.tensor(input_seq), torch.tensor(target)

# (以下為測試程式碼，您可以保留或刪除)
if __name__ == "__main__":
    try:
        train_dataset = StockDataset(f"train_{STOCK_ID}.csv", sequence_length=30)
        test_dataset = StockDataset(f"test_{STOCK_ID}.csv", sequence_length=30)

        print(f"訓練集樣本數: {len(train_dataset)}")
        print(f"測試集樣本數: {len(test_dataset)}")
        sample_input, sample_target = train_dataset[0]
        print(f"輸入序列形狀: {sample_input.shape}")  # 應為 (30, 9)
        print(f"目標形狀: {sample_target.shape}")    # 應為 (9,)
    except FileNotFoundError:
        print("\n請先執行 stock_preprocessing.py 來產生 train_2330.csv 和 test_2330.csv")