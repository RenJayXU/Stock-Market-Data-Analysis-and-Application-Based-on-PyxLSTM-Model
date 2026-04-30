# Stock Market Data Analysis and Application Based on xLSTM Model

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

本專案旨在利用最新提出的 **xLSTM (Extended Long Short-Term Memory)** 架構，針對台灣股市標的（以 0050 成分股為例）進行股價漲跌預測與投資組合應用。本研究深入比較了傳統 LSTM、TCN 與 Transformer 模型，驗證了 xLSTM 在處理具備長時依賴特徵之金融數據時的優越性。

> **致謝：** 本專案之核心 xLSTM 模型架構參考並修改自 [muditbhargava66/PyxLSTM](https://github.com/muditbhargava66/PyxLSTM.git)，並針對股票預測任務進行了特徵工程與回測系統的深度客製化。

---

## 📌 核心特色 (Key Features)

- **先進模型架構**：整合 `mLSTM` 與 `sLSTM` 模組，有效克服傳統 RNN 在金融長序列數據中的梯度消失與記憶瓶頸。
- **深度特徵工程**：系統化建構包含量價、籌碼、動能及趨勢等共 **86 個技術指標**，強化模型輸入特徵的維度。
- **實戰化回測系統**：結合滾動式投資策略（Rolling Strategy），模擬真實交易環境並扣除手續費與交易稅。
- **優異性能表現**：相較於基準模型，xLSTM 在預測準確率（Accuracy）與 F1-Score 上均有顯著提升。

---

## 🛠️ 安裝指南 (Installation)

本專案建議使用 Python 3.10+ 環境。安裝步驟如下：

```bash
# 1. 克隆專案
git clone [https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git](https://github.com/RenJayXU/Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model.git)
cd Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model

# 2. 建立並啟動虛擬環境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安裝依賴套件
pip install -r requirements.txt
