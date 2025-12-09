import torch
import torch.nn as nn
from xLSTM.block import xLSTMBlock

# 【修改】將 input_size 預設改為 9
class StockxLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, 
                 num_blocks=3, dropout=0.4, lstm_type='slstm'):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            xLSTMBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                lstm_type=lstm_type
            ) for _ in range(num_blocks)
        ])
        
        # 【修改】建立 9 個輸出頭
        self.open_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.high_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.low_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.close_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.volume_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.rsi_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.macd_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.sma10_head = nn.Sequential(nn.Linear(hidden_size, 1))
        self.sma20_head = nn.Sequential(nn.Linear(hidden_size, 1))
        
        self._init_weights()

    def _init_weights(self):
        # (權重初始化程式碼不變)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                if 'lstm' in name:
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.0) 
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        for block in self.blocks:
            x, _ = block(x)
        last_hidden = x[:, -1, :]
        
        # 【修改】分別預測 9 個特徵
        open_pred = self.open_head(last_hidden)
        high_pred = self.high_head(last_hidden)
        low_pred = self.low_head(last_hidden)
        close_pred = self.close_head(last_hidden)
        volume_pred = self.volume_head(last_hidden)
        rsi_pred = self.rsi_head(last_hidden)
        macd_pred = self.macd_head(last_hidden)
        sma10_pred = self.sma10_head(last_hidden)
        sma20_pred = self.sma20_head(last_hidden)
        
        # 【修改】合併 9 個預測
        return torch.cat([
            open_pred, high_pred, low_pred, close_pred, volume_pred,
            rsi_pred, macd_pred, sma10_pred, sma20_pred
        ], dim=1)

# (測試函數不變)
def test_initialization():
    model = StockxLSTM(input_size=9, hidden_size=64)
    for name, param in model.named_parameters():
        print(f"參數名: {name}")
        print(f"形狀: {param.shape}")
    print("總輸出形狀 (測試):", model(torch.randn(4, 30, 9))[0].shape)

if __name__ == "__main__":
    test_initialization()