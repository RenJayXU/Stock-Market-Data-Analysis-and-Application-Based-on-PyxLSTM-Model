

```markdown
# Stock Market Data Analysis and Application Based on PyxLSTM Model
# 基於 PyxLSTM (Hybrid xLSTM) 模型的股市數據分析與應用

本專案應用最新的 **xLSTM (Extended Long Short-Term Memory)** 架構於台灣股票市場預測，並針對金融數據特性進行了**混合閘控 (Hybrid Gating)** 機制的改良，解決了數值穩定性問題。最終結合 Markowitz 投資組合理論，提供資產配置建議。

## 🚀 專案亮點 (Key Features)

* **Hybrid xLSTM 架構**：
    * 採用 **mLSTM (Matrix LSTM)** 與 **sLSTM (Scalar LSTM)** 的混合堆疊設計。
    * **改良的混合閘控機制 (Stabilized Gating)**：將遺忘門 (Forget Gate) 改回 Sigmoid，保留輸入門 (Input Gate) 的指數特性，有效解決梯度爆炸與 NaN 問題，同時保有 xLSTM 的強大學習能力。
* **多因子特徵工程**：整合價量資料與技術指標 (RSI, MACD, SMA)。
* **完整的量化評估**：內建 `metrics.py`，自動計算 MSE, MAE, R2, Accuracy (漲跌準確率) 與 F1-Score。
* **基準模型比較**：與傳統 LSTM、TCN (Temporal Convolutional Network)、Transformer 進行效能對比。

## 📂 檔案結構 (File Structure)

```text
.
├── stock_xlstm.py          # [核心] 定義 StockxLSTM 模型 (Hybrid mLSTM+sLSTM)
├── metrics.py              # [核心] 統一的評估指標計算模組
├── stock_preprocessing.py  # 數據預處理 (MinMax Scaling, 技術指標計算)
├── stock_train.py          # 模型訓練腳本 (含 Early Stopping, Gradient Clipping)
├── stock_predict.py        # 預測腳本 (輸出 CSV, 繪圖, 產生 Performance Report)
├── generate_report.py      # 彙整所有模型結果，生成比較圖表與總表
├── portfolio_optimization_final.py # 投資組合優化 (Efficient Frontier)
├── stock_lstm.py           # Benchmark: 傳統 LSTM
├── stock_TCN.py            # Benchmark: TCN
├── stock_transformer.py    # Benchmark: Transformer
└── requirements.txt        # 專案依賴套件

```

##⚡ 快速開始 (Quick Start)###1. 安裝環境```bash
pip install -r requirements.txt

```

###2. 數據預處理讀取原始 CSV 數據，計算技術指標並進行 Min-Max 正規化（範圍 -1 到 1）。

```bash
python stock_preprocessing.py

```

###3. 訓練 Stock-xLSTM 模型使用改良後的 Hybrid 架構進行訓練。

```bash
python stock_train.py

```

###4. 執行預測與評估載入最佳權重，預測未來股價並計算各項指標 (MSE, Accuracy)。

```bash
python stock_predict.py

```

*輸出結果將儲存於 `results/` 資料夾中。*

###5. (可選) 訓練基準模型若要進行模型比較，可執行以下腳本：

```bash
python stock_lstm.py
python stock_TCN.py
python stock_transformer.py

```

###6. 生成比較報告彙整所有模型的表現，繪製成圖表。

```bash
python generate_report.py

```

##📊 實驗結果 (Experimental Results)本研究在台股數據集（如：1301, 2330, 1734 等）上進行了測試。
結果顯示 **Hybrid xLSTM** 在方向預測準確率 (Accuracy) 與擬合度 (R2 Score) 上，相較於傳統 LSTM 有顯著提升。

*詳細數據請參閱 `results/final_model_comparison.csv`。*

##📝 參考文獻 (References)* **Model Core**: Based on [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM).
* **Paper**: Beck, M., et al. (2024). "xLSTM: Extended Long Short-Term Memory". arXiv:2405.04517.

---

**Author**: RenJay Xu

```

```
