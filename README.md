# Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

本專案旨在探索前沿深度學習架構 **xLSTM (Extended Long Short-Term Memory)** 在台灣股市預測與量化交易中的應用。透過整合 0050 成分股的多維技術指標，本研究驗證了 xLSTM 在處理長序列金融數據時，相較於傳統 LSTM、TCN 與 Transformer 模型具備更優異的預測能力與實戰獲利潛力。

## 📌 研究亮點

* **先進模型架構**：實作基於 `mLSTM` 與 `sLSTM` 的 xLSTM 模型，有效解決傳統 RNN 的記憶瓶頸。
* **高維特徵工程**：整合開高低收、籌碼、波動度及動能轉折等 86 種技術指標（共 91 維特徵輸入）。
* **多基準對照**：完整比較 LSTM、TCN 與 Transformer 模型在相同資料集下的表現。
* **實戰量化回測**：結合投資組合優化策略，模擬真實市場交易成本（手續費、稅金），產出實質獲利報告。

---

## 📊 實驗數據與成果

根據專案中的 `generate_report.py` 與 `metrics.py` 評估，xLSTM 模型表現如下：

| 指標 | 數值 | 說明 |
| :--- | :--- | :--- |
| **MSE** | 0.013 | 模型預測值與真實報酬率的均方誤差 |
| **Directional Accuracy** | 55.7% | 預測漲跌方向的準確率 |
| **F1-Score** | 0.636 | 分類預測的綜合評價 |
| **回測收益 (60日)** | +17.64% | 初始 100 萬資金之模擬獲利 |

---

## 🛠️ 技術架構

### 資料來源
* **樣本**：2025年 Q3 之 0050 成分股（半導體、金融、航運等 50 檔）。
* **時間範圍**：2015 年至 2026 年最新交易日。
* **目標變數**：未來 20 天的真實報酬率 (Actual Return)。

### 模型配置
* **Hidden Dimensions**: 128
* **Dropout**: 0.1
* **Learning Rate**: 0.0001
* **Optimizer**: AdamW

---

## 📁 資料夾結構說明

* `stock_xlstm.py`: xLSTM 核心模型架構實作。
* `stock_lstm.py` / `stock_tcn.py`: 基準模型實作。
* `StockDataset.py`: 專門處理時間序列切分與數據清洗的 Dataset 類別。
* `stock_predict.py`: 執行模型推理與產出預測圖表。
* `portfolio_optimization_final.py`: 投資組合回測腳本（含換股邏輯與成本計算）。
* `metrics.py`: 計算各項評估指標。
* `generate_report.py`: 自動化產出 50 檔股票的綜合評估報告與盒鬚圖。

---

## 🚀 快速上手

### 1. 環境安裝
```bash
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
pip install -r requirements.txt
