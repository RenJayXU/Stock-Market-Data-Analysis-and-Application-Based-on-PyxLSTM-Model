## 📁 專案目錄與程式架構 (Repository Structure)

本專案主要分為兩個核心區塊：**改良版 xLSTM 模型** 與 **自建量化評估流程**。

### 1. 模型核心結構 (`xLSTM/`)
此目錄為基於 `PyxLSTM` 延伸修改的底層架構，專為預測連續型金融數據而優化。
* `block.py` / `mlstm.py` / `slstm.py`：xLSTM 神經網路區塊與記憶體擴展機制的改良實作。
* `model.py`：結合上述區塊，建構最終的預測模型。

### 2. 量化交易與分析流程 (`main/`)
此目錄包含本研究獨立開發的端到端（End-to-End）實驗管線：
* **資料獲取與前處理**
  * `fetch_data.py`：自動抓取 2015-01-01 至最新交易日之 50 檔台股歷史數據。
  * `stock_preprocessing.py`：特徵工程，計算 86 種技術指標（涵蓋量能、波動風險、趨勢與動能轉折），並處理極端值防護與反正規化轉換（共 91 維特徵）。
  * `stock_dataset.py`：自定義 PyTorch Dataset，處理 120 天時間步長（Sequence Length）的切窗與特徵對齊。
* **模型訓練與推論**
  * `stock_train.py`：xLSTM 核心訓練腳本。
  * `stock_predict.py`：執行 xLSTM 測試集推論，並輸出預測結果。
* **基準模型比較 (Baselines)**
  * `stock_lstm.py`：LSTM 對照組模型實作與預測。
  * `stock_transformer.py`：Transformer 對照組模型實作與預測。
  * `stock_TCN.py`：TCN 對照組模型實作與預測。
* **評估與量化回測**
  * `metrics.py`：模型效能評估指標計算（MSE, MAE, RMSE, MAPE, R2, Directional Accuracy, F1-Score）。
  * `generate_report.py`：自動彙整 xLSTM 與所有基準模型的評估報告與盒鬚圖（Boxplot），驗證模型泛化穩定度。
  * `portfolio_optimization_final.py`：投資組合回測系統。根據 AI 預測漲幅前 5 名分配權重（各 20%），模擬扣除交易手續費與稅金的真實滾動回測。

---

## 🚀 快速上手 (Quick Start)

### 1. 環境安裝
請確保系統已安裝 Python 3.10+，並執行以下指令安裝所需套件：
```bash
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
pip install -r requirements.txt

cd main

# 步驟 1：下載最新股市資料
python fetch_data.py

# 步驟 2：技術指標計算與資料前處理
python stock_preprocessing.py

# 步驟 3：訓練 xLSTM 模型
python stock_train.py

# 步驟 4：執行 xLSTM 模型預測
python stock_predict.py

# 步驟 5：執行各項基準模型 (Baselines) 進行預測與比較
python stock_lstm.py
python stock_transformer.py
python stock_TCN.py

# 步驟 6：產出綜合績效報告與泛化能力圖表 (包含所有模型比較)
python generate_report.py

# 步驟 7：執行投資組合量化回測 (模擬真實市場交易)
python portfolio_optimization_final.py
