---

## 👨‍💻 作者 (Author)

* **Ren-jie (仁杰)**
* 國立大學 資訊工程學系碩士班 (Department of Computer Science and Information Engineering)
這份更新後的 `README.md` 針對你的碩士論文專案進行了量身打造。內容特別強調了本專案與 `muditbhargava66/PyxLSTM` 的淵源，明確界定你在 `xLSTM` 模組的改良工作，並詳細梳理了你獨立開發的 `main` 目錄完整量化交易流程。

你可以直接複製以下 Markdown 語法並貼上到你的 GitHub `README.md` 檔案中：
```markdown
# Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

本專案為資工所碩士論文之研究實作，旨在探索前沿深度學習架構 **xLSTM (Extended Long Short-Term Memory)** 在台灣股市預測與量化交易中的應用。透過整合 0050 成分股的多維技術指標，本研究驗證了 xLSTM 在處理長序列金融數據時，相較於傳統 LSTM、TCN 與 Transformer 模型具備更優異的預測能力與實戰獲利潛力。

## 📢 專案聲明與致謝 (Acknowledgments)

本專案的核心 xLSTM 模型架構，是基於開源專案 [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM) 進行深度客製化與延伸。

* **模型改良 (`xLSTM/` 目錄)**：本研究僅針對原作者提供的 `xLSTM` 模型結構進行修改與提升，優化 `mLSTM` 與 `sLSTM` 的內部運算機制，使其能夠完美適應高維度、高雜訊的金融時間序列（Stock Market Data）特徵。
* **應用開發 (`main/` 目錄)**：除了底層模型的改良，本專案全新開發了完整的量化交易與模型評估流程（包含資料爬取、特徵工程、多模型基準比較、以及考量交易摩擦成本的動態投資組合回測）。

---

## 📁 核心目錄與程式架構

專案主要分為兩個核心區塊：**改良版 xLSTM 模型** 與 **自建量化評估流程**。

### 1. 模型核心結構 (`xLSTM/`)
此目錄為基於 `PyxLSTM` 延伸修改的底層架構，專為預測連續型金融數據而優化。
* `block.py` / `mlstm.py` / `slstm.py`：xLSTM 神經網路區塊與記憶體擴展機制的改良實作。
* `model.py`：結合上述區塊，建構最終的預測模型。

### 2. 量化交易與分析流程 (`main/`)
此目錄包含本研究獨立開發的端到端（End-to-End）實驗管線：
* `fetch_data.py`：自動抓取 2015-01-01 至最新交易日之 50 檔台股歷史數據。
* `stock_preprocessing.py`：特徵工程，計算 86 種技術指標（涵蓋量能、波動風險、趨勢與動能轉折），並處理極端值防護與反正規化轉換。
* `stock_dataset.py`：自定義 PyTorch Dataset，處理 120 天時間步長（Sequence Length）的切窗與特徵對齊。
* `stock_train.py`：xLSTM 訓練腳本。
* `stock_lstm.py` / `stock_TCN.py` / `stock_transformer.py`：對照組 Benchmark 模型實作。
* `stock_predict.py`：執行測試集推論，並輸出預測結果與真實報酬率之對比。
* `metrics.py`：模型效能評估，計算 MSE、MAE、RMSE、MAPE、R2、Directional Accuracy 及 F1-Score。
* `generate_report.py`：自動彙整 50 檔股票的綜合評估報告與盒鬚圖（Boxplot），驗證模型泛化穩定度。
* `portfolio_optimization_final.py`：投資組合回測系統。根據 AI 預測漲幅前 5 名分配權重，模擬扣除交易手續費與稅金的真實滾動回測。

---

## 📊 實驗數據與成果

根據專案中的 `generate_report.py` 與 `metrics.py` 評估，xLSTM 模型表現如下：

| 指標 | 數值 | 說明 |
| :--- | :--- | :--- |
| **MSE** | 0.013 | 模型預測值與真實報酬率的均方誤差 |
| **Directional Accuracy** | 55.7% | 預測漲跌方向的準確率 |
| **F1-Score** | 0.636 | 分類預測的綜合評價 |
| **回測收益 (60日)** | +17.64% | 初始 100 萬資金，經過 3 次滾動扣除成本後之模擬獲利 |

---

## 🚀 快速上手 (Quick Start)

### 1. 環境安裝
請確保系統已安裝 Python 3.10+，並執行以下指令安裝所需套件：
```bash
git clone https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
pip install -r requirements.txt
