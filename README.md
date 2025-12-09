
````markdown
# 基於 PyxLSTM 模型之股市資料分析及其應用
# Stock Market Data Analysis and Application Based on PyxLSTM Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

本專案利用最新的 **xLSTM (Extended LSTM)** 模型架構，針對台灣股市（TWSE）進行股價趨勢預測，並結合 **Markowitz 現代投資組合理論 (MPT)** 建構最佳化投資策略。

## 📖 專案核心目標 (Project Overview)

本研究旨在解決傳統時間序列模型在金融數據上的限制，並驗證新型架構的實務價值。主要目標歸納為以下四點：

1.  **應用先進模型進行預測**：採用基於 sLSTM 與 mLSTM 區塊的 **xLSTM** 架構，針對台塑 (1301)、台積電 (2330)、國泰金 (2882)、杏國 (1734)、大立光 (3008) 等五支不同特性的股票進行預測，提升長序列時間特徵的捕捉能力。
2.  **模型性能競技與評估**：建立公平的比較框架，將 xLSTM 與主流深度學習模型（**LSTM, TCN, Transformer**）進行對比。評估指標特別強調「**漲跌方向準確率 (Directional Accuracy)**」，以驗證模型捕捉市場趨勢的能力。
3.  **投資組合優化 (Portfolio Optimization)**：這不僅是預測研究，更是應用型專案。我們結合模型的預期回報與歷史風險波動，利用 MPT 理論計算效率前緣 (Efficient Frontier)，提供最大化夏普比率 (Sharpe Ratio) 的資產配置建議。
4.  **驗證新技術的實務價值**：實證結果顯示 xLSTM 在金融時間序列分析中，相比於 Transformer 等現有模型，展現出更優異的穩定性與預測精度。

## 📊 實驗結果摘要 (Benchmark Results)

基於本專案的測試數據（Test Set），xLSTM 在平均表現上優於其他基準模型，特別是在**漲跌方向準確率**與 **R2 Score** 上表現顯著。

### 模型平均性能比較表

| 模型 (Model) | MSE (均方誤差) | MAE (平均絕對誤差) | R2 Score | **Accuracy (漲跌準確率)** |
| :--- | :--- | :--- | :--- | :--- |
| **xLSTM (本專案)** | **2034.01** | **21.53** | **0.935** | **56.18%** |
| Transformer | 10762.47 | 42.41 | 0.632 | 53.70% |
| TCN | 1165.80 | 15.25 | 0.925 | 49.93% |
| LSTM | 10763.73 | 44.77 | 0.588 | 48.04% |

> *數據來源：`results/average_performance.csv`*
> *註：Accuracy 代表模型預測隔日股價漲跌方向的正確率，是交易策略中最關鍵的指標。*

## 📋 專案特點

* **先進架構**：實作 xLSTM (Extended LSTM)，利用指數型門控 (Exponential Gating) 與矩陣記憶體 (Matrix Memory) 解決 LSTM 的長期依賴問題。
* **完整比較**：內建 LSTM、TCN、Transformer 作為 Baseline，程式碼模組化，易於擴充。
* **真實交易邏輯**：除了傳統誤差指標，更關注 Directional Accuracy 與回測績效。
* **自動化報告**：一鍵生成包含 Loss 曲線、股價走勢對比、投資組合效率前緣的完整分析圖表。

## ⚙️ 模型參數設置 (Configuration)

本專案在 `stock_train.py` 中採用的主要超參數設置如下：

* **Sequence Length (Time Steps)**: 30 天
* **Prediction Horizon**: 1 天 (次日預測)
* **Features**: 9 個特徵 (包含 Open, High, Low, Close, Volume, RSI, MACD 等)
* **xLSTM Structure**:
    * Hidden Size: 128
    * Layers: 2
    * Blocks: 2
    * Block Type: sLSTM (Scalar LSTM)
* **Training**:
    * Epochs: 200 (設有 Early Stopping, Patience=20)
    * Batch Size: 64
    * Optimizer: Adam (LR=0.0001)

## 📂 專案結構

```text
├── data/                       # 原始股價 CSV 資料
├── processed_data/             # [自動生成] 預處理後的資料集與 Scaler
├── models/                     # [自動生成] 訓練好的模型權重 (.pth)
├── results/                    # [自動生成] 預測結果 CSV、性能報告與圖表
├── main/                       # 核心程式碼
│   ├── stock_preprocessing.py  # [Step 1] 資料清洗、技術指標計算、標準化
│   ├── stock_train.py          # [Step 2] xLSTM 模型訓練
│   ├── stock_predict.py        # [Step 3] 預測與評估 (產出投資組合所需資料)
│   ├── portfolio_optimization_final.py # [Step 4] 投資組合優化 (Markowitz)
│   ├── generate_report.py      # [Step 5] 產出比較總表 (Metrics Consolidation)
│   ├── stock_xlstm.py          # xLSTM 模型定義
│   ├── stock_dataset.py        # PyTorch Dataset
│   └── Performance Comparison/ # 基準模型 (LSTM, TCN, Transformer)
├── xLSTM/                      # xLSTM 核心模組 (需手動配置)
└── requirements.txt            # Python 依賴
````

## 🛠️ 安裝與環境設定

### 1\. 複製專案

```bash
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
```

### 2\. [關鍵步驟] 配置 xLSTM 依賴

本專案依賴 `muditbhargava66/PyxLSTM` 的實作。**請務必執行以下步驟，否則會出現 `ModuleNotFoundError`**：

1.  下載 [PyxLSTM GitHub](https://github.com/muditbhargava66/PyxLSTM) 專案。
2.  將該專案中的 **`xLSTM` 資料夾** 完整複製到本專案的根目錄下。
3.  確認您的目錄結構中包含：`xLSTM/block.py`, `xLSTM/mlstm.py` 等檔案。

### 3\. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

## 🚀 執行指南 (Execution Guide)

請依照順序執行以下腳本：

**Step 1: 資料前處理**

```bash
python main/stock_preprocessing.py
```

**Step 2: 訓練 xLSTM 模型**

```bash
python main/stock_train.py
```

**Step 3: 產生預測結果**

```bash
python main/stock_predict.py
```

**Step 4: 執行投資組合優化**

```bash
python main/portfolio_optimization_final.py
```

> 此步驟將產出 `results/portfolio_optimization.png`，顯示效率前緣與最佳權重配置。

**Step 5: (選用) 執行基準模型與生成比較報告**
若需重現論文中的比較數據，請執行：

```bash
python "main/Performance Comparison/stock_lstm.py"
python "main/Performance Comparison/stock_TCN.py"
python "main/Performance Comparison/stock_transformer.py"
python main/generate_report.py
```

## 📝 引用與致謝

  * **xLSTM Implementation**: [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM)
  * **Original Paper**: Beck, M., et al. (2024). "xLSTM: Extended Long Short-Term Memory".

-----

**Disclaimer**: 本專案提供的預測結果與投資組合建議僅供學術研究參考，不構成任何實際投資建議。金融市場具有高度不確定性，投資人應自行承擔風險。

```
```
