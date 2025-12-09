````markdown
# Stock Market Data Analysis and Application Based on PyxLSTM Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

本專案旨在利用最新的 **xLSTM (Extended LSTM)** 模型架構進行台灣股市（TWSE）的股價預測，並與傳統深度學習模型（LSTM, TCN, Transformer）進行性能比較。最終結合 **Markowitz 現代投資組合理論 (MPT)**，根據模型預測結果建構最佳化的投資組合策略。

## 📋 專案特點

* **先進模型架構**：採用基於 sLSTM/mLSTM 區塊的 xLSTM 模型，提升長序列時間特徵的捕捉能力。
* **多模型競技**：內建 LSTM、TCN、Transformer 作為基準模型 (Baseline)，並提供公平的比較測試框架。
* **真實交易邏輯**：評估指標包含「漲跌方向準確率 (Directional Accuracy)」，不僅關注價格誤差，更關注交易趨勢。
* **投資組合優化**：整合模型預測回報與歷史風險波動，計算效率前緣 (Efficient Frontier)，提供最大化夏普比率 (Sharpe Ratio) 的資產配置建議。
* **涵蓋標的**：台塑 (1301)、台積電 (2330)、國泰金 (2882)、杏國 (1734)、大立光 (3008)。

## 📂 專案結構

```text
├── data/                       # 原始股價 CSV 資料 (需包含 Date, Open, High, Low, Close, Volume)
├── processed_data/             # [程式自動產生] 預處理後的訓練/驗證/測試集與 Scaler
├── models/                     # [程式自動產生] 訓練好的模型權重檔 (.pth)
├── results/                    # [程式自動產生] 預測結果 CSV、性能報告 txt 與視覺化圖表 png
├── main/                       # 核心程式碼
│   ├── stock_preprocessing.py  # [Step 1] 資料清洗與特徵工程
│   ├── stock_train.py          # [Step 2] xLSTM 模型訓練
│   ├── stock_predict.py        # [Step 3] 預測與評估 (產出投資組合所需資料)
│   ├── portfolio_optimization_final.py # [Step 4] 投資組合優化
│   ├── generate_report.py      # [Step 5] 產出比較總表與圖表 (選用)
│   ├── stock_xlstm.py          # xLSTM 模型定義 (依賴外部 xLSTM 模組)
│   ├── stock_dataset.py        # PyTorch Dataset 定義
│   ├── metrics.py              # 統一評估指標計算
│   └── Performance Comparison/ # 基準模型 (LSTM, TCN, Transformer)
│       ├── stock_lstm.py
│       ├── stock_TCN.py
│       └── stock_transformer.py
├── xLSTM/                      # [重要] xLSTM 核心模組 (需手動配置，見安裝說明)
└── requirements.txt            # Python 套件需求
````

## 🛠️ 安裝與環境設定

### 1\. 複製專案

```bash
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
```

### 2\. [關鍵步驟] 配置 xLSTM 依賴

本專案依賴 `muditbhargava66/PyxLSTM` 的核心實作。由於該套件尚未發布至 PyPI，**您必須確保專案根目錄下包含 `xLSTM` 資料夾**：

1.  下載 [PyxLSTM GitHub](https://github.com/muditbhargava66/PyxLSTM) 專案。
2.  將該專案中的 **`xLSTM` 資料夾** 複製到本專案的根目錄下。
3.  確認目錄結構包含：`xLSTM/block.py`, `xLSTM/mlstm.py` 等檔案。

### 3\. 安裝 Python 套件

請建立 `requirements.txt` 並安裝以下依賴：

```bash
pip install -r requirements.txt
```

**requirements.txt 內容建議：**

```text
torch>=2.0.0
pandas
numpy
matplotlib
seaborn
scikit-learn
ta
joblib
scipy
openpyxl
```

## 🚀 執行指南 (Execution Guide)

請依照以下順序執行程式，以確保資料流正確。

### Step 1: 資料前處理

讀取 `data/` 中的 CSV，計算技術指標 (RSI, MACD)，分割訓練/驗證/測試集，並進行標準化。

```bash
python main/stock_preprocessing.py
```

> **產出**: `processed_data/` 資料夾 (含 `train_xxxx.csv`, `test_xxxx.csv`)。

### Step 2: 訓練 xLSTM 模型

對 5 支股票進行迴圈訓練，並使用早停機制 (Early Stopping) 防止過擬合。

```bash
python main/stock_train.py
```

> **產出**: `models/` 下的 `.pth` 權重檔。

### Step 3: 預測與評估

載入模型對測試集進行預測，產出趨勢準確率與投資組合所需的預測數據。

```bash
python main/stock_predict.py
```

> **產出**:
>
>   * `results/future_prediction_xxxx.csv` (投資組合輸入)
>   * `results/xxxx_prediction.png` (走勢圖)
>   * `results/xxxx_performance.txt` (性能數據)。

### Step 4: 投資組合優化 (應用層)

基於 xLSTM 的預測回報與歷史風險，計算最佳投資權重。

```bash
python main/portfolio_optimization_final.py
```

> **產出**: 最佳權重建議、效率前緣圖 (`results/portfolio_optimization.png`)。

### Step 5: 基準模型比較與報告 (選用)

若需重現論文中的比較實驗，請先執行各個基準模型，再生成報告。

**5-1. 執行基準模型**

```bash
python "main/Performance Comparison/stock_lstm.py"
python "main/Performance Comparison/stock_TCN.py"
python "main/Performance Comparison/stock_transformer.py"
```

**5-2. 生成總表與圖表**

```bash
python main/generate_report.py
```

> **產出**: `model_comparison_chart.png`, `accuracy_by_stock.png`, `final_model_comparison.csv`

## 📊 方法論

### 資料集劃分

為防止 **Data Leakage (資料洩漏)**，我們採用嚴格的時間序列切割：

  * **Training Set (70%)**: 用於模型權重更新。
  * **Validation Set (10%)**: 用於 Early Stopping 監控。
  * **Test Set (20%)**: 用於最終績效評估與投資組合回測 (完全未見過的數據)。

### 評估指標

  * **MSE / MAE**: 衡量價格預測的數值誤差。
  * **Directional Accuracy (Trend)**: 衡量模型判斷漲跌方向的能力。計算邏輯為比較 `(Pred_t - Actual_t-1)` 與 `(Actual_t - Actual_t-1)` 的符號一致性。

### 投資組合策略

使用 **Markowitz Mean-Variance Model**：

  * **預期回報 ($\mu$)**: 來自 xLSTM 對測試集期間的預測年化報酬。
  * **風險矩陣 ($\Sigma$)**: 來自歷史股價的共變異數矩陣 (Covariance Matrix)。
  * **目標**: 最大化夏普比率 (Max Sharpe Ratio)。

## 📝 引用與致謝

  * 本專案的 xLSTM 模型實作參考自 [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM)。
  * xLSTM 原始論文: Beck, M., et al. (2024). "xLSTM: Extended Long Short-Term Memory".

-----

**注意**: 本專案提供的投資組合建議僅供學術研究與回測參考，不構成實際投資建議。投資有風險，入市須謹慎。

```
```
