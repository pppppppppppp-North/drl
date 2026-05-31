import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.bi_lstm import BiLSTMFeatureExtractor

class MultiModalExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Processes 'market' with Bi-LSTM.
    Passes 'news' and 'account' through (or MLP).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        # We need to call super() with the correct shapes
        # The output of this extractor is the concatenated feature vector
        super().__init__(observation_space, features_dim)
        
        # 1. Market Extractor (Bi-LSTM)
        market_space = observation_space.spaces['market']
        self.lstm_out_dim = 32
        self.lstm_extractor = BiLSTMFeatureExtractor(
            market_space, 
            features_dim=self.lstm_out_dim,
            hidden_dim=64,
            num_layers=2
        )
        
        # 2. News/Account concatenation
        # We just flatten/pass them.
        news_dim = observation_space.spaces['news'].shape[0]
        account_dim = observation_space.spaces['account'].shape[0]
        
        total_concat_dim = self.lstm_out_dim + news_dim + account_dim
        
        # Final projection to features_dim
        self.final_fc = nn.Linear(total_concat_dim, features_dim)
        self.activation = nn.ReLU()

    def forward(self, observations: dict) -> torch.Tensor:
        # Extract market features
        market_obs = observations['market'] # Shape: (batch, window, input_dim) or (batch, flattened) depending on SB3 internal
        # SB3 might pass data as it is in the space.
        # But BaseFeaturesExtractor usually expects Tensor.
        # MultiInputPolicy handles key mapping.
        
        lstm_feats = self.lstm_extractor(market_obs)
        
        # Other inputs
        news = observations['news']
        account = observations['account']
        
        # Concatenate
        combined = torch.cat([lstm_feats, news, account], dim=1)
        
        # Project
        out = self.activation(self.final_fc(combined))
        return out
