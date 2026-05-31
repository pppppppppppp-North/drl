import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.models.bi_lstm import BiLSTMFeatureExtractor
from src.models.llm_extractor import BaseLLMExtractor, MockLLMExtractor

class FeaturePipeline:
    """
    Unified feature extraction pipeline that combines:
    1. Bi-LSTM temporal features from market data
    2. LLM contextual features from news (optional)
    """
    
    def __init__(
        self,
        market_feature_dim: int,
        lstm_hidden_dim: int = 64,
        lstm_output_dim: int = 32,
        llm_feature_dim: int = 32,
        sequence_length: int = 30,
        use_llm: bool = False
    ):
        """
        Args:
            market_feature_dim: Number of input features (OHLCV + indicators)
            lstm_hidden_dim: Hidden dimension of Bi-LSTM
            lstm_output_dim: Output dimension of Bi-LSTM features
            llm_feature_dim: Dimension of LLM features
            sequence_length: Number of past timesteps to feed into LSTM
            use_llm: Whether to use LLM features (default: False for now)
        """
        self.sequence_length = sequence_length
        self.use_llm = use_llm
        self.lstm_output_dim = lstm_output_dim
        self.llm_feature_dim = llm_feature_dim
        
        # Initialize Bi-LSTM
        self.bi_lstm = BiLSTMFeatureExtractor(
            input_dim=market_feature_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=lstm_output_dim
        )
        self.bi_lstm.eval()  # Set to evaluation mode
        
        # Initialize LLM (mock for now)
        if use_llm:
            self.llm_extractor: BaseLLMExtractor = MockLLMExtractor(
                embedding_dim=llm_feature_dim
            )
        else:
            self.llm_extractor = None
    
    def preprocess_market_data(self, df: pd.DataFrame, step: int) -> np.ndarray:
        """
        Extract a sequence window of market data ending at 'step'.
        
        Args:
            df: DataFrame with OHLCV + technical indicators
            step: Current timestep
            
        Returns:
            Numpy array of shape (sequence_length, num_features)
        """
        # Define feature columns (exclude non-numeric or meta columns)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Datetime']]
        
        # Get the window
        start_idx = max(0, step - self.sequence_length + 1)
        window_data = df.iloc[start_idx:step+1][feature_cols].values
        
        # Pad if necessary (for early timesteps)
        if len(window_data) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(window_data), window_data.shape[1]))
            window_data = np.vstack([padding, window_data])
        
        return window_data.astype(np.float32)
    
    def extract_lstm_features(self, market_sequence: np.ndarray) -> np.ndarray:
        """
        Extract features using Bi-LSTM.
        
        Args:
            market_sequence: Shape (sequence_length, num_features)
            
        Returns:
            Feature vector of shape (lstm_output_dim,)
        """
        # Convert to torch tensor and add batch dimension
        x = torch.from_numpy(market_sequence).unsqueeze(0)  # (1, seq_len, features)
        
        with torch.no_grad():
            features = self.bi_lstm(x)  # (1, lstm_output_dim)
        
        return features.squeeze(0).numpy()  # (lstm_output_dim,)
    
    def extract_llm_features(self, news_text: Optional[str] = None) -> np.ndarray:
        """
        Extract features using LLM (stub for now).
        
        Args:
            news_text: Optional news text for the current day
            
        Returns:
            Feature vector of shape (llm_feature_dim,)
        """
        if not self.use_llm or self.llm_extractor is None:
            # Return zero vector if LLM is disabled
            return np.zeros(self.llm_feature_dim, dtype=np.float32)
        
        if news_text is None:
            news_text = "No news available"
        
        # Get embedding from LLM
        embedding = self.llm_extractor.get_embedding([news_text])  # (1, llm_feature_dim)
        return embedding.squeeze(0).numpy()
    
    def get_features(
        self,
        df: pd.DataFrame,
        step: int,
        account_state: np.ndarray,
        news_text: Optional[str] = None
    ) -> np.ndarray:
        """
        Main method to extract all features for a given timestep.
        
        Args:
            df: Full DataFrame with market data
            step: Current timestep index
            account_state: Array with [balance, holdings]
            news_text: Optional news for this timestep
            
        Returns:
            Combined feature vector: [LSTM_features, LLM_features, account_state]
        """
        # 1. Extract market sequence and get LSTM features
        market_sequence = self.preprocess_market_data(df, step)
        lstm_features = self.extract_lstm_features(market_sequence)
        
        # 2. Extract LLM features
        llm_features = self.extract_llm_features(news_text)
        
        # 3. Combine all features
        combined = np.concatenate([lstm_features, llm_features, account_state])
        
        return combined.astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """
        Returns the total dimension of the feature vector.
        This is used to set the observation space in the environment.
        """
        # LSTM features + LLM features + account state (2)
        return self.lstm_output_dim + self.llm_feature_dim + 2

if __name__ == "__main__":
    # Test the pipeline
    import pandas as pd
    
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        'Close': np.random.uniform(100, 200, 100),
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000, 10000, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'SMA_10': np.random.uniform(100, 200, 100),
    }, index=dates)
    
    # Initialize pipeline
    pipeline = FeaturePipeline(
        market_feature_dim=7,  # 7 columns in dummy data
        sequence_length=30,
        use_llm=False
    )
    
    # Extract features for step 50
    account_state = np.array([100000.0, 0.0], dtype=np.float32)
    features = pipeline.get_features(df, step=50, account_state=account_state)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Expected dimension: {pipeline.get_feature_dim()}")
    print(f"Feature vector (first 10): {features[:10]}")
