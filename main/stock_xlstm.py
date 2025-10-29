#stock_xlstm
import torch
import torch.nn as nn
from xLSTM.block import xLSTMBlock

class StockxLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, 
                 num_blocks=3, dropout=0.4, lstm_type='slstm'):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 輸入投影層
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # xLSTM區塊堆疊
        self.blocks = nn.ModuleList([
            xLSTMBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                lstm_type=lstm_type
            ) for _ in range(num_blocks)
        ])
        
        # 輸出層結構
        self.open_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 開盤價非負
        )
        self.high_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 最高價非負
        )
        self.low_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 最低價非負
        )
        self.close_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 收盤價非負
        )
        self.volume_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # 成交量非負
        )
        
        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                if 'lstm' in name:
                # 遺忘門偏置初始化為1
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
        
        # 分離預測
        open_pred = self.open_head(last_hidden)
        high_pred = self.high_head(last_hidden)
        low_pred = self.low_head(last_hidden)
        close_pred = self.close_head(last_hidden)
        volume_pred = self.volume_head(last_hidden)
        
        return torch.cat([open_pred, high_pred, low_pred, close_pred, volume_pred], dim=1)

def test_initialization():
    """獨立測試函數"""
    model = StockxLSTM(input_size=5, hidden_size=64)
    
    # 檢查參數維度
    for name, param in model.named_parameters():
        print(f"參數名: {name}")
        print(f"形狀: {param.shape}")
        print(f"數值範圍: {param.data.min():.4f} ~ {param.data.max():.4f}")
        print("-"*50)

if __name__ == "__main__":
    # 僅在直接執行時運行測試
    test_initialization()