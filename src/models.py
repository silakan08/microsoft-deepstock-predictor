import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) PyTorch Model for Time Series Regression.
    
    Architecture:
    - Stacked LSTM layers (num_layers=2) with hidden units (e.g. 32 or 50)
    - Fully Connected (Linear) Output Layer to predict single continuous price value
    """
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Recurrent Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Readout Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state h0 and cell state c0 with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) PyTorch Model for Time Series Regression.
    
    Architecture:
    - Stacked GRU layers (num_layers=2) with hidden units (e.g. 32 or 50)
    - Fully Connected (Linear) Output Layer to predict single continuous price value
    - Note: GRUs use update and reset gates (no separate cell state memory), making them faster to train.
    """
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Recurrent Layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Readout Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state h0 with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        # out shape: (batch_size, seq_length, hidden_dim)
        out, hn = self.gru(x, h0)
        
        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
