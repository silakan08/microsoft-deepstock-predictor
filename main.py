import os
import torch
from src.fetch_data import fetch_stock_data
from src.data_loader import load_and_preprocess_data
from src.models import LSTMModel, GRUModel
from src.train import train_model
from src.evaluate import evaluate_and_compare

def main():
    print("=" * 60)
    print("  MICROSOFT SUMMER INTERNSHIP - ML STOCK PREDICTION PIPELINE  ")
    print("=" * 60)
    
    # Adım 1: Veri Setini Çekme (Amazon AMZN Stock Data 2012-2024)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "AMZN_stock_data.csv")
    
    if not os.path.exists(csv_path):
        print("\n[STEP 1] Fetching Stock Data from Yahoo Finance API...")
        fetch_stock_data(ticker="AMZN", start_date="2012-01-01", end_date="2024-01-01", output_dir=data_dir)
    else:
        print(f"\n[STEP 1] Found existing stock dataset at '{csv_path}'.")
        
    # Adım 2: Veri Ön İşleme (MinMaxScaler [-1,1], Sliding Window lookback=20, %80 Train / %20 Test)
    print("\n[STEP 2] Preprocessing Data & Generating Sliding Window Sequences...")
    data_dict = load_and_preprocess_data(csv_path, lookback=20, train_ratio=0.8)
    
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    
    # Adım 3: PyTorch LSTM Modelini Eğitme
    print("\n[STEP 3] Initializing & Training PyTorch LSTM Model...")
    lstm_model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2)
    print("LSTM Architecture:")
    print(lstm_model)
    lstm_tr_loss, lstm_val_loss, lstm_time = train_model(lstm_model, X_train, y_train, val_ratio=0.1, epochs=60, lr=0.001, batch_size=32)
    
    # Adım 4: PyTorch GRU Modelini Eğitme
    print("\n[STEP 4] Initializing & Training PyTorch GRU Model...")
    gru_model = GRUModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2)
    print("GRU Architecture:")
    print(gru_model)
    gru_tr_loss, gru_val_loss, gru_time = train_model(gru_model, X_train, y_train, val_ratio=0.1, epochs=60, lr=0.001, batch_size=32)
    
    # Adım 5: Değerlendirme, Backtesting & Metrik Karşılaştırması
    print("\n[STEP 5] Evaluating Models & Running Backtesting Simulation...")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    metrics, payload = evaluate_and_compare(
        lstm_model, gru_model, data_dict, 
        lstm_tr_loss, lstm_val_loss, lstm_time,
        gru_tr_loss, gru_val_loss, gru_time,
        output_dir=results_dir
    )
    
    # Modelleri kaydetme
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(lstm_model.state_dict(), os.path.join(models_dir, "lstm_model.pth"))
    torch.save(gru_model.state_dict(), os.path.join(models_dir, "gru_model.pth"))
    print(f"[SUCCESS] Trained model weights saved to '{models_dir}'.")

if __name__ == "__main__":
    main()
