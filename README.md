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
1. 克隆倉庫

git clone https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git

cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

2. 導入 xLSTM 核心模組

由於 xLSTM 核心實作基於 muditbhargava66 的工作並進行了大量定制，您需要先下載原始模組，然後再應用我們的自訂修改。

克隆或下載原始 PyxLSTM 倉庫：

git clone https://github.com/muditbhargava66/PyxLSTM.git

將下載的 PyxLSTM 倉庫中的 xLSTM 資料夾複製到您的 Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model 根目錄。

請將針對本專案所做的修改（例如，對 mlstm.py、slstm.py 和 block.py 的變更）套用到 xLSTM 資料夾中，方法是將對應的檔案取代或更新為本倉庫中提供的檔案。確保目錄結構類似於 YourRepository/xLSTM/block.py。

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
```
## ⚙️ 實驗設計與參數設定 (Experimental Setup)

*   **資料集**：2025 年 Q3 台灣 50 指數（0050）成分股，共 50 檔股票。
*   **時間範圍**：2015-01-01 至 2026-04-26。
*   **特徵工程**：收錄開高低收等基本指標，並結合籌碼量能、波動風險、趨勢判定及動能轉折等，共計 86 個技術指標（輸入特徵共 91 維）。
*   **預測目標**：未來 20 個交易日之真實報酬率 (`Actual_Return`)。
*   **模型超參數**：
    *   Sequence Length: 120
    *   Hidden Size: 128
    *   Num Blocks: 2 
    *   Dropout: 0.1
    *   Learning Rate: 0.0001

---

## 📊 實驗數據與基準比較 (Model Evaluation)

本研究比較了 xLSTM 與目前主流的時間序列預測模型。實驗結果顯示，經過改良的 xLSTM 在降低誤差與方向預測準確率上，皆具備顯著優勢：

| Model | MSE | RMSE | R2 | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | 0.0232 | 0.1321 | -1.6515 | 49.39% | 0.4447 
| **TCN** | 0.0206 | 0.1246 | -1.2853 | 49.98% | 0.4714 
| **Transformer**| 0.0230 | 0.1221 | -0.9497 | 49.75% | 0.4810 
| **xLSTM (Ours)**| **0.0130** | **0.0965** | **-0.1324** | **55.71%** | **0.6364** 

---

## 🏆 實戰量化回測 (Quantitative Backtesting)

為驗證模型的實戰價值，本專案模擬真實投資情境，將預測結果轉化為實際的投資組合，並與大盤（0050 ETF）進行終極複利對決。

*   **策略設計**：初始資金 1,000,000 元，每期挑選 xLSTM 預測看漲分數最高的前 5 檔股票，進行等權重（各 20%）配置。
*   **回測設定**：每 20 個交易日為一期，共執行 3 次滾動調倉，並**真實扣除所有交易手續費與稅金**。

**回測結果總結**：
*   **累積摩擦成本**：6,871 元
*   **xLSTM 組合 60 日最終淨值**：1,176,401 元（累積報酬 **+17.64%**）
*   **0050 ETF 60 日最終淨值**：1,204,802 元（累積報酬 **+20.48%**）
