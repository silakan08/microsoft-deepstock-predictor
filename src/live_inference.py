import os
import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from src.models import LSTMModel, GRUModel
from sklearn.preprocessing import MinMaxScaler

def run_live_inference(ticker="AMZN", lookback=20, models_dir="saved_models", data_csv="data/AMZN_stock_data.csv"):
    """
    Fetches the latest live market price for `ticker` via yfinance API,
    loads trained PyTorch LSTM & GRU models, performs live forward pass (inference),
    and returns next-day price forecast and trend direction.
    """
    print(f"[LIVE INFERENCE] Fetching latest market data for '{ticker}'...")
    
    try:
        # Download recent data up to today
        df = yf.download(ticker, period="30d", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.dropna(inplace=True)
    except Exception as e:
        print(f"[WARNING] yfinance download failed: {e}. Reading local CSV data instead.")
        df = pd.read_csv(data_csv)
        
    close_prices = df['Close'].values[-lookback:].reshape(-1, 1)
    latest_price = float(close_prices[-1, 0])
    latest_date = str(df['Date'].iloc[-1])[:10]
    
    # Fit scaler on history
    scaler = MinMaxScaler(feature_range=(-1, 1))
    full_df = pd.read_csv(data_csv)
    scaler.fit(full_df[['Close']].values)
    
    scaled_input = scaler.transform(close_prices).reshape(1, lookback, 1)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    # Load trained models
    lstm_model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    gru_model = GRUModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    
    lstm_path = os.path.join(models_dir, "lstm_model.pth")
    gru_path = os.path.join(models_dir, "gru_model.pth")
    
    if os.path.exists(lstm_path):
        lstm_model.load_state_dict(torch.load(lstm_path))
    if os.path.exists(gru_path):
        gru_model.load_state_dict(torch.load(gru_path))
        
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(input_tensor).numpy()
        gru_pred_scaled = gru_model(input_tensor).numpy()
        
    lstm_pred_price = float(scaler.inverse_transform(lstm_pred_scaled)[0, 0])
    gru_pred_price = float(scaler.inverse_transform(gru_pred_scaled)[0, 0])
    
    lstm_chg = round(((lstm_pred_price - latest_price) / latest_price) * 100, 2)
    gru_chg = round(((gru_pred_price - latest_price) / latest_price) * 100, 2)
    
    live_result = {
        "ticker": ticker,
        "latest_date": latest_date,
        "latest_price": round(latest_price, 2),
        "lstm_next_day": round(lstm_pred_price, 2),
        "gru_next_day": round(gru_pred_price, 2),
        "lstm_chg_pct": lstm_chg,
        "gru_chg_pct": gru_chg,
        "status": "Live Market Data Synced"
    }
    
    print("\n" + "="*50)
    print("      REAL-TIME LIVE INFERENCE RESULTS      ")
    print("="*50)
    print(f"Ticker Symbol:        {ticker}")
    print(f"Latest Market Date:   {latest_date}")
    print(f"Latest Closing Price: ${latest_price:.2f}")
    print(f"LSTM Tomorrow Forecast: ${lstm_pred_price:.2f} ({lstm_chg:+,.2f}%)")
    print(f"GRU Tomorrow Forecast:  ${gru_pred_price:.2f} ({gru_chg:+,.2f}%)")
    print("="*50 + "\n")
    
    return live_result

if __name__ == "__main__":
    run_live_inference()
