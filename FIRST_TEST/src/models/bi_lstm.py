import torch
import torch.nn as nn

class BiLSTMFeatureExtractor(nn.Module):
    """
    Bi-Directional LSTM for extracting temporal features from market data.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 32):
        super(BiLSTMFeatureExtractor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bi-LSTM Layer
        # batch_first=True expects input shape: (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layer to project LSTM output to desired feature dimension
        # Input to FC is hidden_dim * 2 (because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        # LSTM output shape: (batch, seq_len, num_directions * hidden_size)
        # h_n, c_n shapes: (num_layers * num_directions, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # We take the output of the last time step for feature extraction
        # shape: (batch, hidden_dim * 2)
        last_time_step = out[:, -1, :]
        
        # Project to feature space
        features = self.activation(self.fc(last_time_step))
        
        return features

if __name__ == "__main__":
    # Test
    batch_size = 4
    seq_len = 30
    input_dim = 12 # OHLCV + Indicators
    model = BiLSTMFeatureExtractor(input_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (4, 32)
