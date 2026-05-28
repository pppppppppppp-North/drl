import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BiLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Bi-Directional LSTM for extracting temporal features from market data.
    Compatible with Stable Baselines3.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32, num_layers: int = 2, hidden_dim: int = 64):
        # We assume the observation space is flattened: (batch, seq_len * input_dim) 
        # OR we handle the shaping logic carefully.
        # SB3 passes flattened observations by default for MlpPolicy.
        # So we need to reconstruct (batch, seq_len, input_dim).
        # Let's assume input_dim is derived from args or fixed.
        # For simplicity, let's say the environment passes a 1D vector which IS the feature vector if we were simple.
        # But here we want the sequence. 
        # To do this correctly with SB3 MlpPolicy, we often use a custom policy or a Dict observation space.
        
        # Strategy: The Env will provide a buffer of the last N steps as observation.
        # Observation Shape: (Window_Size, Input_Dim) flattened -> (Window_Size * Input_Dim)
        
        super(BiLSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Infer dimensions
        # We need the user to tell us window_size and input_channels, or infer from shape total.
        # Let's assume input_channels (OHLCV+Ind) is known, say 12.
        self.input_dim = 12 # OHLCV (5) + Indicators (7) approx
        self.seq_len = observation_space.shape[0] // self.input_dim
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, features_dim)
        self.activation = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch_size, seq_len * input_dim)
        """
        # Reshape to (batch, seq_len, input_dim)
        batch_size = observations.shape[0]
        x = observations.view(batch_size, -1, self.input_dim)
        
        # LSTM
        out, (h_n, c_n) = self.lstm(x)
        
        # Take last time step
        last_time_step = out[:, -1, :]
        
        # Project
        features = self.activation(self.fc(last_time_step))
        
        return features

if __name__ == "__main__":
    # Test
    seq_len = 30
    input_dim = 12
    obs_shape = (seq_len * input_dim,)
    space = gym.spaces.Box(low=-1, high=1, shape=obs_shape)
    
    model = BiLSTMFeatureExtractor(space, features_dim=32)
    dummy_input = torch.randn(4, seq_len * input_dim)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 

