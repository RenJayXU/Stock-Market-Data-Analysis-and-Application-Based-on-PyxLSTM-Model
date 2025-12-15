```markdown
# 基於 PyxLSTM 模型之股市資料分析及其應用
# Stock Market Data Analysis and Application Based on PyxLSTM Model

這是一個完整的金融科技（FinTech）實作專案，旨在利用最新的深度學習架構 **xLSTM (Extended LSTM)** 來預測台灣股市個股走勢，並基於預測結果結合現代投資組合理論進行資產配置優化。

本專案不僅探討了 xLSTM 在金融時間序列上的適用性，更將其與傳統 LSTM、TCN 及 Transformer 等主流模型進行了嚴謹的效能對比。

## 📖 專案目錄
- [專案簡介](#-專案簡介)
- [核心特色](#-核心特色)
- [環境需求](#-環境需求)
- [檔案結構說明](#-檔案結構說明)
- [使用說明](#-使用說明)
- [實驗結果](#-實驗結果)

## 🚀 專案簡介

本研究的核心目標是驗證新一代 **xLSTM** 架構（包含 sLSTM 與 mLSTM）的矩陣記憶（Matrix Memory）與指數門控（Exponential Gating）機制，是否能比傳統序列模型更有效地捕捉股價的長短期依賴關係。

專案涵蓋了從**資料清洗**、**特徵工程**、**模型訓練**、**效能評估**到**投資組合優化**的完整流程，並針對台股熱門標的（如台積電 2330、國泰金 2882 等）進行實證分析。

## ✨ 核心特色

1.  **前沿模型應用**：實作並應用最新的 xLSTM 深度學習模型於金融預測。
2.  **多模型基準對比**：提供公平的基準測試，包含：
    -   **LSTM** (Long Short-Term Memory)
    -   **TCN** (Temporal Convolutional Network)
    -   **Transformer** (Attention Mechanism)
3.  **完整的 FinTech 流程**：
    -   自動化計算技術指標（MA, RSI 等）。
    -   滑動視窗（Sliding Window）資料集建構。
    -   正規化與反正規化處理。
4.  **投資組合應用**：基於模型預測結果，利用 Markowitz Efficiency Frontier 進行投資組合權重優化，提供具體的交易決策支援。
5.  **視覺化報告**：自動生成 Loss 收斂圖、股價預測對比圖及模型效能比較圖。

## 🛠 環境需求

本專案基於 Python 開發，主要依賴 `torch` 及 `pyxlstm` 函式庫。

請參考 `requirements.txt` 進行安裝：

```bash
pip install -r requirements.txt

```

*注意：xLSTM 相關依賴可能需要特定的 CUDA 版本支援，請參閱 [pyxlstm](https://github.com/NX-AI/xlstm) 官方文檔。*

##📂 檔案結構說明本專案程式碼結構分為四大模組，詳細用途如下：

###A. 資料處理 (Data Processing)* **`stock_preprocessing.py`**: 負責原始股價資料清洗、特徵選取（Open, High, Low, Close, Volume）、技術指標計算及資料正規化。
* **`stock_dataset.py`**: 定義 PyTorch Dataset，實作滑動視窗機制，將時間序列轉換為模型可讀的 Tensor 格式。

###B. 模型建構 (Model Architecture)* **`stock_xlstm.py`**: **核心檔案**。定義 xLSTM 模型架構（整合 sLSTM/mLSTM 區塊）。
* **`stock_lstm.py`**: 定義傳統 LSTM 模型（Baseline）。
* **`stock_TCN.py`**: 定義時間卷積網路 TCN 模型（Baseline）。
* **`stock_transformer.py`**: 定義 Transformer 模型（Baseline）。

###C. 訓練與預測 (Training & Prediction)* **`stock_train.py`**: 執行模型訓練流程。包含超參數設定、Loss 監控、早停機制（Early Stopping）及最佳模型權重儲存。
* **`stock_predict.py`**: 載入訓練好的模型權重進行推論，並將預測結果反正規化，輸出預測 CSV。

###D. 評估與應用 (Evaluation & Application)* **`metrics.py`**: 提供 RMSE, MAE, MAPE, R2 Score 等評估指標計算函式。
* **`generate_report.py`**: 生成分析報告，繪製 Loss 曲線、預測對比圖及模型比較長條圖。
* **`portfolio_optimization_final.py`**: **最終應用**。利用預測結果計算預期報酬，執行投資組合優化，輸出最佳資產配置權重。

##⚡ 使用說明 (Usage)###1. 資料前處理讀取原始 CSV 資料並進行標準化與切割：

```bash
python stock_preprocessing.py

```

###2. 模型訓練執行訓練腳本（可於程式碼中切換 `model_type` 選擇 xLSTM, LSTM, TCN 或 Transformer）：

```bash
python stock_train.py

```

###3. 執行預測載入權重並對測試集進行預測：

```bash
python stock_predict.py

```

###4. 產生評估報告計算誤差指標並繪製圖表：

```bash
python generate_report.py

```

###5. 投資組合優化根據預測結果進行資產配置分析：

```bash
python portfolio_optimization_final.py

```

##📊 實驗結果 (Project Results)執行完畢後，您將在 `results/` 資料夾中看到以下產出：

* **股價預測圖** (`*_prediction.png`): 直觀展示預測線與真實股價的貼合程度。
* **Loss 曲線** (`*_loss.png`): 展示模型訓練收斂情況。
* **量化評估報告** (`average_performance.csv`): 包含 RMSE, MAPE 等精確誤差數據，證明 xLSTM 在特定情境下的優越性。
* **模型比較圖** (`model_comparison_chart.png`): 各模型效能長條圖對比。
* **投資組合建議** (`portfolio_optimization.png`): 展示在風險可控下的最佳資產配置權重（例如：台積電 30%、國泰金 20%...）。

---

**Disclaimer**: 本專案僅供學術研究與技術交流使用，不構成任何投資建議。投資有風險，入市須謹慎。

```

```
