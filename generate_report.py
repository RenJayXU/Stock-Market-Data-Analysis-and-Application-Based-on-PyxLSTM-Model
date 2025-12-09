import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# 設定圖表風格
plt.style.use('ggplot')
sns.set_palette("husl")

RESULTS_DIR = "results"
STOCKS = ['1301', '2330', '2882', '1734', '3008']

def parse_xlstm_results():
    """讀取 stock_predict.py 產生的 txt 檔案"""
    data = []
    for stock in STOCKS:
        path = os.path.join(RESULTS_DIR, f"{stock}_performance.txt")
        if not os.path.exists(path):
            print(f"[警告] 找不到 xLSTM 結果: {path}")
            continue
            
        entry = {'Stock': stock, 'Model': 'xLSTM'}
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':')
                    key = key.strip()
                    try:
                        entry[key] = float(val)
                    except ValueError:
                        pass # 忽略非數值欄位
        
        # 修正欄位名稱以對齊 CSV
        if 'Trend Accuracy' in entry: # 假如 txt 寫的是 Trend Accuracy
            entry['Accuracy'] = entry.pop('Trend Accuracy')
        if 'F1 Score' in entry:
            entry['F1'] = entry.pop('F1 Score')
            
        data.append(entry)
    
    return pd.DataFrame(data)

def load_benchmark_csv(filename, model_name):
    """讀取比較模型的 CSV"""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"[警告] 找不到 {model_name} 結果: {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    # 確保 Model 欄位正確
    df['Model'] = model_name 
    return df

def main():
    print("=== 開始彙整分析報告 ===")
    
    # 1. 蒐集所有資料
    df_xlstm = parse_xlstm_results()
    df_lstm = load_benchmark_csv("lstm_benchmark.csv", "LSTM")
    df_tcn = load_benchmark_csv("tcn_benchmark.csv", "TCN")
    df_trans = load_benchmark_csv("transformer_benchmark.csv", "Transformer")
    
    # 合併
    all_results = pd.concat([df_xlstm, df_lstm, df_tcn, df_trans], ignore_index=True)
    
    if all_results.empty:
        print("錯誤：沒有讀取到任何結果資料。請先執行各個模型的訓練腳本。")
        return

    # 儲存總表
    final_csv_path = os.path.join(RESULTS_DIR, "final_model_comparison.csv")
    all_results.to_csv(final_csv_path, index=False)
    print(f"\n已儲存詳細比較表: {final_csv_path}")
    
    # 2. 計算平均表現 (Mean Performance)
    # 只選取數值欄位進行平均
    numeric_cols = ['MSE', 'MAE', 'R2', 'Accuracy', 'F1']
    avg_results = all_results.groupby('Model')[numeric_cols].mean().reset_index()
    
    print("\n=== 各模型平均表現 (5檔股票平均) ===")
    print(avg_results.round(4))
    
    avg_csv_path = os.path.join(RESULTS_DIR, "average_performance.csv")
    avg_results.to_csv(avg_csv_path, index=False)

    # 3. 視覺化比較圖
    print("\n正在繪製比較圖表...")
    
    metrics_to_plot = ['MSE', 'MAE', 'Accuracy', 'F1']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        # 繪製長條圖
        sns.barplot(data=all_results, x='Model', y=metric, hue='Model', dodge=False)
        
        plt.title(f'Comparison of {metric}')
        plt.xlabel('')
        plt.ylabel(metric)
        
        # 標示數值 (平均值)
        # 為了簡潔，我們只在圖上畫出分佈，數值請參考 CSV
        
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    print(f"圖表已儲存: {chart_path}")
    
    # 4. 額外：畫各別股票的 Accuracy 比較
    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_results, x='Stock', y='Accuracy', hue='Model')
    plt.title('Directional Accuracy by Stock')
    plt.ylim(0, 1.0) # 準確率通常在 0~1 之間
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    acc_chart_path = os.path.join(RESULTS_DIR, "accuracy_by_stock.png")
    plt.savefig(acc_chart_path, dpi=300)
    print(f"各股準確率圖表已儲存: {acc_chart_path}")

    print("\n=== 全部完成！ ===")
    print("您現在可以查看 results 資料夾中的圖表與 CSV 來撰寫論文了。")

if __name__ == "__main__":
    main()