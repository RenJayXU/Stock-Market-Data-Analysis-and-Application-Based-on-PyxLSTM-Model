# Stock-Market-Data-Analysis-and-Application-Based-on-PyxLSTM-Model
Stock Market Data Analysis 
執行順序與使用說明 (Execution Guide)
本專案包含從資料前處理、模型訓練、性能比較到投資組合優化的完整流程。請依照以下順序執行各個 Python 腳本。

1. 資料前處理 (Data Preprocessing)
這是所有模型共用的基礎，必須最先執行。

執行檔案: main/stock_preprocessing.py

功能: 讀取原始股價 CSV (data/)，計算技術指標 (RSI, MACD)，分割訓練/驗證/測試集，並進行標準化。

產出: processed_data/ 資料夾 (含 train_xxxx.csv, test_xxxx.csv, xxxx_scaler.save)。

2. 訓練 xLSTM 模型 (Train xLSTM Model)
訓練並評估您的核心預測模型。

執行檔案: main/stock_train.py

功能: 讀取處理後的資料，訓練 xLSTM 模型，並在驗證集上執行早停 (Early Stopping)。

產出: models/ 資料夾 (含 xxxx_best_model.pth 權重檔)。

3. 模型預測與評估 (Prediction & Evaluation)
利用訓練好的模型對測試集進行預測，並產出投資組合所需的數據。

執行檔案: main/stock_predict.py

功能: 載入訓練好的模型，對測試集進行預測。計算準確率並繪製走勢圖。

產出:

results/future_prediction_xxxx.csv (投資組合優化的輸入檔)

results/xxxx_performance.txt (比較報表的數據來源)

results/xxxx_prediction.png (股價走勢圖)

4. 基準模型比較 (Benchmark Comparison) - 可選
若需重現論文中的比較實驗，請執行以下腳本（順序不拘）。它們使用與 xLSTM 相同的資料集。

執行檔案:

main/Performance Comparison/stock_lstm.py (LSTM 模型)

main/Performance Comparison/stock_TCN.py (TCN 模型)

main/Performance Comparison/stock_transformer.py (Transformer 模型)

產出: results/ 資料夾下的各模型 benchmark CSV 檔。

5. 產出分析報告 (Generate Report) - 可選
彙整 xLSTM 與其他基準模型的表現，生成比較圖表。

執行檔案: main/generate_report.py

功能: 讀取所有模型的評估結果，畫出長條比較圖。

產出: results/final_model_comparison.csv (總表)、model_comparison_chart.png (比較圖)。

6. 投資組合優化 (Portfolio Optimization)
最後一步，將 AI 預測轉化為投資策略。

執行檔案: main/portfolio_optimization_final.py

功能: 讀取 stock_predict.py 產生的預測檔，結合歷史風險計算 Markowitz 效率前緣。

產出: 最佳投資權重建議、效率前緣圖表 (portfolio_optimization.png)。

核心模組 (Core Modules)
以下檔案為系統核心組件，由上述程式呼叫，請勿直接執行或刪除：

main/stock_dataset.py: 負責將 CSV 轉換為 PyTorch Tensor 的資料集類別。

main/metrics.py: 定義統一的評估指標（如漲跌準確率）計算函數。

main/stock_xlstm.py: 定義 xLSTM 神經網路架構。
