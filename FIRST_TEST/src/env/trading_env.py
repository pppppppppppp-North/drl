import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from src.models.feature_pipeline import FeaturePipeline
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 100000.0, 
        transaction_fee: float = 0.001,
        use_feature_pipeline: bool = True,
        sequence_length: int = 30
    ):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.use_feature_pipeline = use_feature_pipeline
        
        self.action_space = spaces.Discrete(3)
        
        if use_feature_pipeline:
            feature_cols = [col for col in df.columns if col not in ['Date', 'Datetime']]
            market_feature_dim = len(feature_cols)
            
            self.feature_pipeline = FeaturePipeline(
                market_feature_dim=market_feature_dim,
                lstm_output_dim=32,
                llm_feature_dim=32,
                sequence_length=sequence_length,
                use_llm=False
            )
            self.feature_dim = self.feature_pipeline.get_feature_dim()
        else:
            self.feature_pipeline = None
            self.feature_dim = 66
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )
        
        self.balance = initial_balance
        self.holdings = 0
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.prev_portfolio_value = initial_balance  # Track for reward calculation
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = 0
        self.prev_portfolio_value = self.initial_balance
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute Action
        if action == 1:  # Buy
            max_shares = self.balance // (current_price * (1 + self.transaction_fee))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.holdings += max_shares
                
        elif action == 2:  # Sell
            if self.holdings > 0:
                revenue = self.holdings * current_price * (1 - self.transaction_fee)
                self.balance += revenue
                self.holdings = 0
        
        # Step forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Calculate NEW portfolio value
        new_price = self.df.iloc[self.current_step]['Close'] if not terminated else current_price
        portfolio_value = self.balance + (self.holdings * new_price)
        
        # FIXED REWARD: Percentage return (scaled to reasonable range)
        reward = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        reward = np.clip(reward * 100, -10, 10)  # Scale and clip to ±10
        
        self.prev_portfolio_value = portfolio_value
        
        obs = self._get_observation()
        info = {'portfolio_value': portfolio_value}
        
        return obs, reward, terminated, truncated, info
    def _get_observation(self) -> np.ndarray:
        account_state = np.array([self.balance, self.holdings], dtype=np.float32)
        
        if self.use_feature_pipeline and self.feature_pipeline is not None:
            features = self.feature_pipeline.get_features(
                df=self.df,
                step=self.current_step,
                account_state=account_state,
                news_text=None
            )
        else:
            random_features = np.random.randn(self.feature_dim - 2).astype(np.float32)
            features = np.concatenate([random_features, account_state])
        
        return features
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}")
