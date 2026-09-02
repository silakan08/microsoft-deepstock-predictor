import time
import os
import torch
import torch.nn as nn
import numpy as np

def train_model(model, X_train, y_train, val_ratio=0.1, epochs=60, lr=0.001, batch_size=32):
    """
    Trains PyTorch model (LSTM or GRU) with Validation Split for Overfitting & Loss Tracking.
    Returns train_loss_history, val_loss_history, and training_time.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train / Val Split for Loss Curve Tracking
    val_size = int(len(X_train) * val_ratio)
    train_size = len(X_train) - val_size
    
    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]
    
    train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss_history = []
    val_loss_history = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= train_size
        train_loss_history.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_loss_history.append(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:03d}/{epochs:03d}] -> Train Loss: {epoch_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
    training_time = time.time() - start_time
    print(f"[TRAINING COMPLETE] Time taken: {training_time:.2f} seconds")
    
    return train_loss_history, val_loss_history, training_time
