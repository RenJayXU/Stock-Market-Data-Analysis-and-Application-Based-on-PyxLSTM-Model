import torch
import torch.nn as nn
from xLSTM.block import xLSTMBlock

class StockxLSTM(nn.Module):
    def __init__(self, input_size=91, hidden_size=128, num_layers=1, 
                 num_blocks=2, dropout=0.4, lstm_type=['mlstm', 'slstm']): 
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size # 紀錄 input_size，用來對齊最後的輸出
        
        # 1. 輸入投影層 (將特徵投射到 hidden_size 維度)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)  # 強制將輸入特徵的變異數壓制，防止後續指數門控暴衝
        )
        
        # 2. 建構 xLSTM Blocks (支援混合模式)
        self.blocks = nn.ModuleList()
        
        if isinstance(lstm_type, str):
            lstm_types = [lstm_type] * num_blocks
        else:
            if len(lstm_type) != num_blocks:
                print(f"警告: lstm_type 列表長度 ({len(lstm_type)}) 與 num_blocks ({num_blocks}) 不符，將自動循環補齊。")
                import itertools
                lstm_types = list(itertools.islice(itertools.cycle(lstm_type), num_blocks))
            else:
                lstm_types = lstm_type

        for i in range(num_blocks):
            self.blocks.append(
                xLSTMBlock(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    lstm_type=lstm_types[i] 
                )
            )
        
        # 🔴 3. 輸出層：【關鍵修改】廢除 9 頭設計，改為單一動態輸出層
        # 這樣不管 input_size 是 9 還是 35，輸出都會自動對齊！
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 傳統 LSTM 會設為 1.0，但 xLSTM 必須設為 0.0 或更小
                nn.init.constant_(param, 0.0) 
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden=None):
        # x shape: (Batch, Seq_Len, Input_Size)
        x = self.input_proj(x)
        
        # 通過每一層 xLSTM Block
        for block in self.blocks:
            x, _ = block(x) 
            
        last_hidden = x[:, -1, :]
        
        # 🔴 動態輸出預測結果
        # 如果輸入是 35 維，這裡就會輸出 35 維
        output = self.output_layer(last_hidden)
        
        return output

if __name__ == "__main__":
    # 測試 9 維度
    print("--- 測試 9 維度架構 ---")
    model_9 = StockxLSTM(input_size=9, hidden_size=64, num_blocks=2, lstm_type=['mlstm', 'slstm'])
    test_input_9 = torch.randn(4, 30, 9)
    output_9 = model_9(test_input_9)
    print("輸出形狀 (應為 4, 9):", output_9.shape) 
    
    # 測試 35 維度
    print("\n--- 測試 35 維度架構 (壓力測試) ---")
    model_35 = StockxLSTM(input_size=35, hidden_size=64, num_blocks=2, lstm_type=['mlstm', 'slstm'])
    test_input_35 = torch.randn(4, 30, 35)
    output_35 = model_35(test_input_35)
    print("輸出形狀 (應為 4, 35):", output_35.shape)