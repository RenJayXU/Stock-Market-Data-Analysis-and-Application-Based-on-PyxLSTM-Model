import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入自定義模組
from stock_xlstm import StockxLSTM
from stock_dataset import StockDataset
from metrics import calculate_metrics

# ==========================================
# 設定參數 (必須與 stock_train.py 完全一致！)
# ==========================================
CONFIG = {
    "sequence_length": 120,   # 【重要修正】對齊 train 的 40
    "prediction_days": 1,
    "hidden_size": 128,      
    "num_layers": 1,         
    "num_blocks": 2,         
    "dropout": 0.1,          # 【重要修正】對齊 train 的 0.2
    "lstm_type": ['mlstm', 'slstm'] 
}

# 完整 50 檔成份股
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
    '1301', '1303', '6505', '1216', '2002', '9910', '9921', '1402'
]

def predict_stock(stock_id, device):
    # 1. 載入 Scaler 
    scaler_path = f"processed_data/{stock_id}_scaler.save"
    if not os.path.exists(scaler_path):
        return None
    scaler = joblib.load(scaler_path)

    # 2. 準備測試資料集
    test_csv = f"processed_data/test_{stock_id}.csv"
    if not os.path.exists(test_csv):
        return None
        
    df_test_raw = pd.read_csv(test_csv, index_col=0)
    test_dates = pd.to_datetime(df_test_raw.index)

    test_dataset = StockDataset(
        test_csv,
        sequence_length=CONFIG["sequence_length"],
        prediction_days=CONFIG["prediction_days"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 載入模型
    model = StockxLSTM(
        input_size=91,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_blocks=CONFIG["num_blocks"],
        dropout=CONFIG["dropout"],
        lstm_type=CONFIG["lstm_type"]
    ).to(device)
    
    model_path = f"models/{stock_id}_best_model.pth"
    if not os.path.exists(model_path):
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

# 4. 執行推論
    pred_return, actual_return, prev_actuals = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            
            # 🔴 核心修正：
            # 1. 確保 input 形狀是 91 維 (防止測試集也發生 92 維錯誤)
            if inputs.shape[-1] > 91:
                inputs = inputs[:, :, :-1]
                
            # 直接取輸出層的第 0 個節點作為預測報酬率
            pred_return.append(output[0, 0].cpu().numpy())
            actual_return.append(targets[0].numpy())
            
            # 為了給 metrics 計算參考，保留最後一天的特徵狀態
            prev_actuals.append(inputs[0, -1, :].cpu().numpy())

    pred_return = np.array(pred_return)
    actual_return = np.array(actual_return)

    # 5. 反正規化 
    # (🔴 只需要還原 inputs 特徵，用來抓取昨天的收盤價或特徵，預測值不需要還原)
    prev_inverse = scaler.inverse_transform(np.array(prev_actuals))
    
    # 這裡的 prev_return 只是給 metrics.py 用作 Baseline (比如昨天漲跌多少)
    # 取決於你在 preprocessing 時 Return 放哪個位置，如果是第四個就是 [:, 3]
    prev_return = prev_inverse[:, 3] 

    valid_length = len(pred_return)
    plot_dates = test_dates[-valid_length:]


    # 6. 計算績效指標
    metrics = calculate_metrics(actual_return, pred_return, prev_return)
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/{stock_id}_performance.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 儲存預測結果 CSV (供 MPT 投資組合優化使用)
    result_df = pd.DataFrame({
        "Date": plot_dates,
        "Actual_Return": actual_return,
        "Predicted_Return": pred_return
    })
    result_df.to_csv(f"results/future_prediction_{stock_id}.csv", index=False)
    
    # ==========================================
    # 7. 繪圖 (必須放在所有變數都計算完之後！)
    # ==========================================
    # 挑選具代表性的股票畫圖，避免產生 50 張圖
    if stock_id in ['2330', '2882', '2603', '1301', '3231']: 
        plt.figure(figsize=(12, 6))
        plt.plot(plot_dates, actual_return, label="Actual Return", color='blue', alpha=0.7)
        plt.plot(plot_dates, pred_return, label="Predicted Return (xLSTM)", color='red', linestyle='--', alpha=0.9)
        plt.title(f"{stock_id} Return Prediction (Out-of-Sample)\nAcc: {metrics['Accuracy']:.2%}, F1: {metrics['F1']:.4f}")
        plt.xlabel("Date")
        plt.ylabel("Return Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/{stock_id}_prediction.png", dpi=300)
        plt.close()

    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_stock_metrics = []

    # 迴圈執行預測
    for stock in tqdm(STOCK_LIST, desc="正在分析成份股預測績效"):
        metrics = predict_stock(stock, device)
        if metrics is not None:
            metrics['Stock'] = stock
            all_stock_metrics.append(metrics)

    # 產出總表與板塊分析
    if all_stock_metrics:
        df_summary = pd.DataFrame(all_stock_metrics)
        df_summary.to_csv("results/all_stocks_metrics_detail.csv", index=False)
        
        # --- 定義產業板塊 ---
        SECTOR_MAPPING = {
            'Semiconductor': ['2330', '2454', '3711', '2303', '2344', '2449', '3037', '3008', '2408'],
            'Financial': ['2891', '2881', '2882', '2886', '2884', '2892', '2890', '5880', '2883', '2887', '2885', '5871', '2880'],
            'Electronic & AI': ['2317', '2382', '2412', '3045', '4904', '3231', '2357', '4938', '2301', '2356', '2345', '2376', '2368', '6669', '2395'],
            'Traditional & Shipping': ['1216', '2002', '1301', '1303', '1326', '1101', '1102', '2603', '2609', '2610', '2618']
        }
        
        def get_sector(stock_id):
            for sector, stocks in SECTOR_MAPPING.items():
                if str(stock_id) in stocks: return sector
            return 'Other'
            
        df_summary['Sector'] = df_summary['Stock'].apply(get_sector)

        # --- 產業板塊效能表 ---
        sector_group = df_summary.groupby('Sector')[['Accuracy', 'F1', 'R2', 'MSE']].mean().reset_index()
        
        print("\n" + "="*30)
        print("各產業板塊平均表現 (Sector Analysis)")
        print(sector_group.to_string(index=False))

        # --- 極端表現個股分析 (Top 3 & Bottom 3) ---
        df_sorted = df_summary.sort_values(by='Accuracy', ascending=False)
        
        print("\n" + "="*30)
        print("表現最佳前 3 檔 (Top 3 by Accuracy):")
        for _, row in df_sorted.head(3).iterrows():
            print(f"Stock: {row['Stock']} | Accuracy: {row['Accuracy']:.2%} | F1: {row['F1']:.4f} | R2: {row['R2']:.4f}")
            
        print("\n表現最差後 3 檔 (Bottom 3 by Accuracy):")
        for _, row in df_sorted.tail(3).iterrows():
            print(f"Stock: {row['Stock']} | Accuracy: {row['Accuracy']:.2%} | F1: {row['F1']:.4f} | R2: {row['R2']:.4f}")

        # --- 總體表現 ---
        avg_metrics = df_summary[['MSE', 'MAE', 'RMSE', 'MAPE', 'R2', 'Accuracy', 'F1']].mean()
        print("\n" + "="*30)
        print("0050 ETF 預測績效綜合報告 (xLSTM)")
        print("-" * 30)
        print(f"分析股票總數: {len(df_summary)} 檔")
        print(f"平均 MSE:      {avg_metrics['MSE']:.4f}")
        print(f"平均 MAE:      {avg_metrics['MAE']:.4f}")
        print(f"平均 R2:       {avg_metrics['R2']:.4f}")
        print(f"平均 Accuracy: {avg_metrics['Accuracy']:.2%}")
        print(f"平均 F1-Score: {avg_metrics['F1']:.4f}")
        print("="*30)

if __name__ == "__main__":
    main()