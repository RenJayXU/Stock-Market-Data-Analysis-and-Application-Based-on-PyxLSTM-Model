import torch
import pandas as pd
import numpy as np
import joblib
from stock_xlstm import StockxLSTM
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt

def predict_future_close(num_days=60):
    scaler = joblib.load("2330_scaler.save")
    rawdata = pd.read_csv("2330.csv")
    rawdata = rawdata.sort_values('Date')
    last_date = pd.to_datetime(rawdata['Date'].iloc[-1])

    features = ["Open", "High", "Low", "Close", "Volume"]
    featuredata = rawdata[features]
    processeddata = scaler.transform(featuredata)

    model = StockxLSTM(input_size=5, hidden_size=64, num_layers=2, num_blocks=3)
    model.load_state_dict(torch.load("2330model.pth"))
    model.eval()

    input_sequence = processeddata[-30:].copy()
    predictions = []
    current_date = last_date

    with torch.no_grad():
        for i in range(num_days):
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
            pred = model(input_tensor).cpu().numpy()[0]  # shape: (5,)
            dummy = np.zeros((1, 5))
            dummy[0, 3] = pred[3]  # 只保留Close
            real_pred = scaler.inverse_transform(dummy)[0]
            close_pred = real_pred[3]

            pred_date = (current_date + BDay(1)).strftime('%Y-%m-%d')
            predictions.append((pred_date, close_pred))
            current_date += BDay(1)

            # 更新序列，只更新Close，其它特徵維持不變
            new_day = input_sequence[-1].copy()
            new_day[3] = pred[3]
            input_sequence = np.vstack([input_sequence[1:], new_day])

    print("\n未來{}天Close預測：".format(num_days))
    print("=================================")
    print("Date        Close")
    for date, close in predictions:
        print(f"{date}  {close:8.2f}")
    print("=================================")

    # 畫出未來10天的收盤價走勢
    pred_dates = [x[0] for x in predictions]
    close_prices = [x[1] for x in predictions]
    plt.figure(figsize=(10, 6))
    plt.plot(pred_dates, close_prices, marker='o', linestyle='-', color='b')
    plt.title('Predicted Close Price for Next 60 Days')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Predicted Close Price for Next 60 Days.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    df = pd.DataFrame(predictions, columns=["Date", "Predicted_Close"])
    df.to_excel("future_prediction2330.xlsx", index=False)
    print("已儲存預測為 future_prediction.xlsx")

if __name__ == "__main__":
    predict_future_close(num_days=60)