import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_stock_data(ticker="AMZN", start_date="2012-01-01", end_date=None, output_dir="data"):
    """
    Downloads historical stock price data for the specified ticker up to today's date.
    Saves the cleaned dataset as a CSV file.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_stock_data.csv")
    
    print(f"[INFO] Fetching historical stock data for '{ticker}' from {start_date} to {end_date}...")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError("yfinance returned empty dataframe")
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
    except Exception as e:
        print(f"[WARNING] yfinance download failed/returned empty: {e}. Trying direct fallback source...")
        fallback_url = "https://stooq.com/q/d/l/?s=amzn.us&i=d"
        df = pd.read_csv(fallback_url)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date)].sort_values('Date').reset_index(drop=True)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    df.dropna(inplace=True)
    
    df.to_csv(file_path, index=False)
    print(f"[SUCCESS] Dataset successfully saved to '{file_path}' ({len(df)} rows).")
    return df

if __name__ == "__main__":
    fetch_stock_data()
