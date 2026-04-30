Stock Market Data Analysis and Application Based on xLSTM Model本專案旨在利用最新的 xLSTM (Extended Long Short-Term Memory) 架構，針對台灣股市標的（0050 成分股）進行股價漲跌預測與投資組合應用。本研究特別比較了傳統 LSTM、TCN 與 Transformer 模型，驗證了 xLSTM 在長序列金融數據中的優越性。註： 本專案之核心 xLSTM 實作參考並修改自 muditbhargava66/PyxLSTM。📌 核心亮點先進模型架構：整合 mLSTM 與 sLSTM 模組，解決傳統循環神經網路在處理長時間依賴關係時的效能瓶頸。豐富的特徵工程：計算包含量價、籌碼、動能及趨勢等共 86 個技術指標（Technical Indicators）。實戰回測系統：不僅限於準確度評估，更包含滾動式投資組合模擬，考量真實交易成本（手續費與稅金）。優異的預測能力：在 0050 成分股的實驗中，預測準確率（Accuracy）達到 55.71%，F1-Score 達 0.636。🛠️ 安裝指南 (Installation)由於本專案基於 pyxlstm 進行開發，安裝步驟與原版相似，並加入了額外的技術指標分析工具。Bash# 克隆專案
git clone https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

# 建立虛擬環境 (建議)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安裝依賴套件
pip install -r requirements.txt
主要依賴： torch, pyxlstm, yfinance, pandas_ta, scikit-learn, matplotlib📊 資料集與特徵 (Dataset & Features)研究對象：2025 年 Q3 之台灣 50 指數 (0050) 全部成分股。資料期間：2015/01/01 至 2026/04/26。預測目標：預測未來 T+20 天之收盤價變動方向與幅度。特徵工程：共 86 個特徵，包含但不限於：趨勢指標：SMA, EMA, ADX動能指標：RSI, MACD, Stochastic波動與量能：Bollinger Bands, ATR, OBV自定義插件：針對台股特性優化之技術特徵。🏗️ 模型配置 (Model Configuration)本研究使用的 xLSTM 模型超參數設定如下：參數 (Hyperparameter)設定值 (Value)Sequence Length (Look-back)120 daysHidden Size128Num Layers1Num Blocks2LSTM Typemlstm / slstmLearning Rate0.0001Dropout0.1📈 實驗結果 (Experimental Results)在與基準模型的對比測試中，xLSTM 在所有指標上均表現最佳：模型 (Model)MSE ↓MAE ↓RMSE ↓R2 ↑Accuracy ↑F1-Score ↑LSTM0.02320.10600.1321-1.651549.39%0.4447TCN0.02060.10050.1246-1.285349.98%0.4714Transformer0.02300.09870.1221-0.949749.75%0.4810xLSTM0.01300.07570.0965-0.132455.71%0.6364💰 投資模擬與應用 (Trading Simulation)我們實施了滾動式投資組合策略，每期選取預測漲幅最高的 5 支標的進行等權重配置：模擬期間：2026 年初至 2026/04/26（共 60 個交易日）。策略表現：xLSTM 投資組合：累計報酬率 +17.64%。0050 ETF (基準)：累計報酬率 +20.48%。成本考量：計算結果已扣除約 6,871 元 之交易稅與手續費，展現了模型在真實市場中的穩健獲利潛力與泛化能力。📂 檔案結構 (Project Structure)Plaintext├── data/                       # 存放下載的原始資料
├── pyxlstm/                    # 核心模型架構 (基於 PyxLSTM 修改)
├── fetch_data.py               # 資料抓取腳本 (yfinance)
├── stock_preprocessing.py      # 特徵工程與 86 個技術指標計算
├── train.py                    # 模型訓練主程式
├── metrics.py                  # 效能評估指標實作
├── portfolio_optimization.py   # 投資組合策略與回測模擬
├── requirements.txt            # 專案依賴清單
└── README.md
📜 參考文獻與致謝Beck, M., Pöppel, K., et al. (2024). xLSTM: Extended Long Short-Term Memory.PyxLSTM 原始實作：muditbhargava66/PyxLSTM本研究係基於上述框架進行修改，並新增了針對股市數據的特徵工程、損失函數優化與回測系統。
