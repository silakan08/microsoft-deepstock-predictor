import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    """PyTorch Dataset for Time Series Stock Sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(csv_path, lookback=20, train_ratio=0.8):
    """
    Loads stock CSV data, normalizes 'Close' prices to [-1, 1],
    creates sliding window sequences of length `lookback`,
    and splits into sequential Train/Test sets as PyTorch Tensors.
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    close_prices = df[['Close']].values
    
    # 1. MinMaxScaler (-1 to 1) as requested in specification
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # 2. Sliding Window Sequence Creation (Lookback = N past days to predict Day N+1)
    X, y = [], []
    for i in range(len(scaled_prices) - lookback):
        X.append(scaled_prices[i : i + lookback])
        y.append(scaled_prices[i + lookback])
        
    X = np.array(X) # Shape: (samples, lookback, 1)
    y = np.array(y) # Shape: (samples, 1)
    
    # 3. Sequential Train/Test Split (80% train, 20% test)
    train_size = int(len(X) * train_ratio)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Dates corresponding to test targets
    test_dates = df['Date'].values[lookback + train_size:]
    
    print(f"[DATA PREPARATION]")
    print(f"  Total sequences: {len(X)}")
    print(f"  Train set: {X_train.shape[0]} sequences")
    print(f"  Test set:  {X_test.shape[0]} sequences")
    print(f"  Lookback window: {lookback} days")
    
    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "scaler": scaler,
        "df": df,
        "test_dates": test_dates
    }
