import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定圖表風格
plt.style.use('ggplot')
sns.set_palette("husl")

RESULTS_DIR = "results"

STOCK_LIST = [
    # 前 10 大權重股
    '2330', '2317', '2454', '2308', '2382', '3711', '2303', '2891', '2881', '2882',
    
    # 金融控股族群
    '2886', '2884', '2892', '2890', '5880', '2883', '2887', '2885', '5871', '2880', '5876',
    
    # 電子、半導體與 AI 供應鏈 (含 Q3 新增: 2059)
    '2412', '3045', '4904', '3231', '2357', '2059', '2301', '2345', '3008', '2379', 
    '2408', '3034', '2327', '2474', '2395', '6669',
    
    # 航運與交通 (含 Q1 新增: 2615)
    '2603', '2609', '2615', '2207',
    
    # 生技醫療 (Q3 新增: 6919)
    '6919',
    
    # 塑化、鋼鐵、食品與傳產 (已剔除: 1326, 1101, 1760)
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402']
# 🔴 統一模型排序：傳統基準模型在前，xLSTM 壓軸
MODEL_ORDER = ['LSTM', 'TCN', 'Transformer', 'xLSTM']
# 🔴 給予 xLSTM 特別的顏色 (例如紅色系)，其餘用冷色系
MODEL_COLORS = {'LSTM': '#1f77b4', 'TCN': '#2ca02c', 'Transformer': '#ff7f0e', 'xLSTM': '#d62728'}

def load_xlstm_results():
    detail_csv = os.path.join(RESULTS_DIR, "all_stocks_metrics_detail.csv")
    if os.path.exists(detail_csv):
        df = pd.read_csv(detail_csv)
        df['Model'] = 'xLSTM'
        return df
    
    data = []
    for stock in STOCK_LIST:
        path = os.path.join(RESULTS_DIR, f"{stock}_performance.txt")
        if not os.path.exists(path):
            continue
        entry = {'Stock': stock, 'Model': 'xLSTM'}
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':')
                    entry[key.strip()] = float(val)
        data.append(entry)
    return pd.DataFrame(data)

def load_benchmark_csv(filename, model_name):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['Model'] = model_name
    return df

def plot_model_comparison(df_summary):
    """圖表 C: 專注於 Accuracy 的高對比度比較圖"""
    if 'Accuracy' not in df_summary.columns:
        return
        
    plt.figure(figsize=(10, 6))
    # 🔴 加入 order 與 custom palette 確保 xLSTM 顏色最突出
    sns.barplot(x='Model', y='Accuracy', data=df_summary, 
                order=MODEL_ORDER, palette=MODEL_COLORS, capsize=.1)
    
    plt.title('Average Accuracy Comparison Across Models (50 Stocks)', fontsize=14, fontweight='bold')
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Random Guess (50%)')
    plt.ylabel('Directional Accuracy')
    
    max_acc = df_summary.groupby('Model')['Accuracy'].mean().max()
    plt.ylim(0.45, max(0.80, max_acc + 0.1)) 
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Model_Accuracy_Comparison_Focus.png'), dpi=300)
    plt.close()

def plot_boxplot_distribution(df_summary):
    """🔴 新增圖表 D: 50檔股票的 Accuracy 盒鬚圖 (展示泛化穩定度)"""
    if 'Accuracy' not in df_summary.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Accuracy', data=df_summary, order=MODEL_ORDER, palette=MODEL_COLORS)
    # 加上散佈點，可以清楚看到每檔股票的具體落點
    sns.stripplot(x='Model', y='Accuracy', data=df_summary, order=MODEL_ORDER, 
                  color='black', alpha=0.3, jitter=True)
                  
    plt.title('Accuracy Distribution Across 50 Stocks (Robustness Check)', fontsize=14, fontweight='bold')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random Guess (50%)')
    plt.ylabel('Directional Accuracy')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Model_Accuracy_Boxplot.png'), dpi=300)
    plt.close()

def main():
    print("=== 開始彙整 0050 成份股分析報告 ===")
    
    df_xlstm = load_xlstm_results()
    df_lstm = load_benchmark_csv("lstm_benchmark.csv", "LSTM")
    df_tcn = load_benchmark_csv("tcn_benchmark.csv", "TCN")
    df_trans = load_benchmark_csv("transformer_benchmark.csv", "Transformer")
    
    all_results = pd.concat([df_xlstm, df_lstm, df_tcn, df_trans], ignore_index=True)
    
    if all_results.empty:
        print("錯誤：沒有讀取到任何資料。")
        return

    # 過濾出有效的模型並確保按照指定順序排序
    present_models = [m for m in MODEL_ORDER if m in all_results['Model'].unique()]
    all_results['Model'] = pd.Categorical(all_results['Model'], categories=present_models, ordered=True)

    all_results.to_csv(os.path.join(RESULTS_DIR, "final_model_comparison_0050.csv"), index=False)
    
    metrics_cols = ['MSE', 'MAE', 'RMSE', 'R2', 'Accuracy', 'F1']
    available_metrics = [c for c in metrics_cols if c in all_results.columns]
    avg_results = all_results.groupby('Model')[available_metrics].mean().reset_index()
    
    print("\n" + "="*50)
    print("各模型在 50 檔成份股的平均表現 (Final Comparison)")
    print("-" * 50)
    print(avg_results.round(4).to_string(index=False))
    print("="*50)
    
    avg_results.to_csv(os.path.join(RESULTS_DIR, "average_performance_comparison.csv"), index=False)

    print("\n正在生成論文圖表...")
    
    # 🔴 圖表 A: 將 MAE 替換為 F1-Score，並固定模型排序與顏色
    plot_metrics = ['Accuracy', 'F1', 'R2', 'MSE'] 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(plot_metrics):
        if metric in all_results.columns:
            sns.barplot(data=all_results, x='Model', y=metric, ax=axes[i], 
                        order=present_models, palette=MODEL_COLORS, capsize=.1)
            axes[i].set_title(f'Mean {metric} across 0050 Stocks', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('')
            if metric in ['Accuracy', 'F1']:
                axes[i].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
            
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_summary.png"), dpi=300)
    plt.close()

    # 圖表 B: 50 檔股票的 Accuracy 遍歷圖 (已稍微調寬以容納 200 根柱子)
    if 'Accuracy' in all_results.columns:
        plt.figure(figsize=(24, 8))
        sns.barplot(data=all_results, x='Stock', y='Accuracy', hue='Model', 
                    hue_order=present_models, palette=MODEL_COLORS)
        plt.axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)')
        plt.title('Directional Accuracy for Each 0050 Stock', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        
        max_acc_all = all_results['Accuracy'].max()
        plt.ylim(0.45, max(0.85, max_acc_all + 0.05)) 
        
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "accuracy_by_stock_0050.png"), dpi=300)
        plt.close()

    # 圖表 C & D: 生成焦點長條圖與盒鬚圖
    plot_model_comparison(all_results)
    plot_boxplot_distribution(all_results)

    print(f"\n全部完成！請至 {RESULTS_DIR} 查看相關 CSV 與圖表。")

if __name__ == "__main__":
    main()