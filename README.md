```markdown
# Stock Market Data Analysis and Application Based on PyxLSTM Model
# 基於 PyxLSTM 模型的股市數據分析與應用

本專案旨在探討並應用最新的 **xLSTM (Extended Long Short-Term Memory)** 架構於台灣股票市場的價格預測與投資組合優化。

本研究的核心模型代碼基於 [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM.git) 提供的 xLSTM 實現進行延伸開發，針對金融時間序列數據的特性進行了客製化修改，構建了適合股市分析的 **Stock-xLSTM** 模型，並進一步結合現代投資組合理論 (Markowitz Portfolio Optimization) 進行資產配置建議。

## 專案特點 (Key Features)

1.  **基於 xLSTM 的核心架構**：
    * 引用並改進了 xLSTM (sLSTM/mLSTM) 的區塊設計，利用其指數型閘控 (Exponential Gating) 與矩陣記憶體 (Matrix Memory) 特性，提升對長序列股價數據的特徵捕捉能力。
    * **多頭輸出層設計 (Multi-Head Output)**：不同於傳統僅預測收盤價的模型，本專案的 `StockxLSTM` 模型修改了輸出層，同時預測 9 個關鍵特徵（Open, High, Low, Close, Volume, RSI, MACD, SMA10, SMA20），透過多任務學習提升模型的泛化能力與穩定性。

2.  **完整的量化交易流程**：
    * **數據預處理**：包含技術指標計算 (RSI, MACD, MA) 與正規化。
    * **模型訓練**：針對台股特定個股（如：台積電 2330, 台塑 1301 等）進行獨立模型訓練。
    * **績效評估**：與傳統 LSTM、TCN、Transformer 等模型進行基準比較。

3.  **投資組合優化應用**：
    * 利用 xLSTM 預測的未來收益率作為輸入，結合歷史波動率矩陣。
    * 使用 **Markowitz 效率前緣 (Efficient Frontier)** 模型，計算在特定風險下的最大夏普比率 (Max Sharpe Ratio) 投資組合權重。

## 專案結構 (Directory Structure)

```text
.
├── models/                     # 存放訓練好的模型權重 (.pth)
├── processed_data/             # 存放預處理後的訓練與測試數據 (.csv, .save)
├── results/                    # 存放損失曲線圖、預測結果圖與績效報告
├── xLSTM/                      # (引用自 PyxLSTM) xLSTM 核心模組
├── stock_xlstm.py              # [核心] 客製化的 StockxLSTM 模型架構定義
├── stock_train.py              # 模型訓練腳本 (包含 Training Loop 與 Validation)
├── stock_predict.py            # 模型預測腳本 (產生未來股價預測)
├── stock_preprocessing.py      # 數據預處理與技術指標計算
├── stock_dataset.py            # PyTorch Dataset 定義
├── portfolio_optimization_final.py # 投資組合優化與效率前緣繪製
├── requirements.txt            # 專案依賴套件
└── README.md                   # 專案說明文件

```

##安裝 (Installation)1. **複製專案**
```bash
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

```


2. **安裝依賴套件**
建議使用 Python 3.8 以上版本。
```bash
pip install -r requirements.txt

```



##使用方法 (Usage)本專案針對五檔代表性台股進行分析：1301 (台塑), 1734 (杏輝), 2330 (台積電), 2882 (國泰金), 3008 (大立光)。

###步驟 1: 數據預處理讀取原始股價數據，計算技術指標並進行 Min-Max 正規化。

```bash
python stock_preprocessing.py

```

###步驟 2: 訓練模型執行訓練腳本，模型會自動訓練並儲存最佳權重至 `models/` 資料夾，並繪製 Loss 曲線至 `results/`。

```bash
python stock_train.py

```

*可在 `stock_train.py` 中調整超參數 (Hyperparameters)，如 `sequence_length`, `hidden_size`, `num_epochs` 等。*

###步驟 3: 執行預測載入訓練好的模型，對測試集或未來數據進行預測，並輸出預測結果 CSV。

```bash
python stock_predict.py

```

###步驟 4: 投資組合優化讀取各個股的預測結果，計算預期報酬與風險矩陣，並繪製效率前緣圖，輸出最佳資產配置建議。

```bash
python portfolio_optimization_final.py

```

##致謝與參考 (Acknowledgements & References)本專案的模型核心架構參考並改寫自以下開源專案：

* **PyxLSTM**: [https://github.com/muditbhargava66/PyxLSTM.git](https://github.com/muditbhargava66/PyxLSTM.git)
* 感謝 muditbhargava66 提供的 xLSTM (Extended LSTM) 基礎實作。本研究在此基礎上修改了 `block.py` 與模型輸出層，以適配多變數回歸（Multi-variate Regression）的股市預測任務。

