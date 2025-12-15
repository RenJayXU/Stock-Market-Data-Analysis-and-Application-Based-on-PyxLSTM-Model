import torch
import torch.nn as nn
from xLSTM.block import xLSTMBlock

class StockxLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, 
                 num_blocks=2, dropout=0.4, lstm_type=['mlstm', 'slstm']): 
        # 修改：lstm_type 預設改為列表，代表第一塊用 mLSTM，第二塊用 sLSTM
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. 輸入投影層 (將 9 個特徵投射到 hidden_size 維度)
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 2. 建構 xLSTM Blocks (支援混合模式)
        self.blocks = nn.ModuleList()
        
        # 如果使用者傳入的是單一字串 (例如 "slstm")，我們就把它複製成列表
        if isinstance(lstm_type, str):
            lstm_types = [lstm_type] * num_blocks
        else:
            # 如果傳入的是列表，確保長度跟 num_blocks 一樣，不夠就循環補齊
            if len(lstm_type) != num_blocks:
                print(f"警告: lstm_type 列表長度 ({len(lstm_type)}) 與 num_blocks ({num_blocks}) 不符，將自動循環補齊。")
                import itertools
                lstm_types = list(itertools.islice(itertools.cycle(lstm_type), num_blocks))
            else:
                lstm_types = lstm_type

        print(f"模型架構配置: {lstm_types}") # 印出確認目前的架構

        for i in range(num_blocks):
            self.blocks.append(
                xLSTMBlock(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    lstm_type=lstm_types[i] # 每一層使用指定的類型
                )
            )
        
        # 3. 輸出層 (保持原本的 9 頭設計)
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
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 【絕對關鍵修正】
                # 傳統 LSTM 會設為 1.0，但 xLSTM 必須設為 0.0 或更小
                # 否則 e^1.0 = 2.7，記憶會指數爆炸
                nn.init.constant_(param, 0.0) 
                
                # 請確認下面這段程式碼已經被【刪除】或【註解】掉：
                # if 'lstm' in name:
                #     n = param.size(0)
                #     start, end = n//4, n//2
                #     param.data[start:end].fill_(1.0)  <-- 這行是 NaN 的元兇
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden=None):
        # x shape: (Batch, Seq_Len, Input_Size)
        x = self.input_proj(x)
        
        # 通過每一層 xLSTM Block
        for block in self.blocks:
            x, _ = block(x) # 這裡的 block 會根據我們初始化的類型自動切換 sLSTM 或 mLSTM
            
        last_hidden = x[:, -1, :]
        
        # 多頭輸出預測
        open_pred = self.open_head(last_hidden)
        high_pred = self.high_head(last_hidden)
        low_pred = self.low_head(last_hidden)
        close_pred = self.close_head(last_hidden)
        volume_pred = self.volume_head(last_hidden)
        rsi_pred = self.rsi_head(last_hidden)
        macd_pred = self.macd_head(last_hidden)
        sma10_pred = self.sma10_head(last_hidden)
        sma20_pred = self.sma20_head(last_hidden)
        
        return torch.cat([
            open_pred, high_pred, low_pred, close_pred, volume_pred,
            rsi_pred, macd_pred, sma10_pred, sma20_pred
        ], dim=1)

if __name__ == "__main__":
    # 測試混合架構
    print("--- 測試 Hybrid 架構 (mLSTM -> sLSTM) ---")
    model = StockxLSTM(input_size=9, hidden_size=64, num_blocks=2, lstm_type=['mlstm', 'slstm'])
    test_input = torch.randn(4, 30, 9)
    output = model(test_input)
    print("輸出形狀:", output.shape) # 預期 (4, 9)