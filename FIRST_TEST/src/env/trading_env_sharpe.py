import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from src.models.feature_pipeline import FeaturePipeline

class TradingEnvSharpe(gym.Env):
    """Trading environment with Sharpe Ratio-based rewards."""
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 100000.0, 
        transaction_fee: float = 0.001,
        use_feature_pipeline: bool = True,
        sequence_length: int = 30,
        reward_window: int = 20  # Window for Sharpe calculation
    ):
        super(TradingEnvSharpe, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.use_feature_pipeline = use_feature_pipeline
        self.reward_window = reward_window
        
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
        
        # Track returns for Sharpe calculation
        self.returns_history: List[float] = []
        self.prev_portfolio_value = initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = 0
        self.returns_history = []
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
        
        # Calculate portfolio value
        new_price = self.df.iloc[self.current_step]['Close'] if not terminated else current_price
        portfolio_value = self.balance + (self.holdings * new_price)
        
        # Calculate return for this step
        step_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(step_return)
        self.prev_portfolio_value = portfolio_value
        
        # Calculate Sharpe Ratio reward
        reward = self._calculate_sharpe_reward()
        
        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
            'sharpe_reward': reward,
            'step_return': step_return
        }
        
        return obs, reward, terminated, truncated, info

    def _calculate_sharpe_reward(self) -> float:
        """Calculate rolling Sharpe Ratio as reward."""
        if len(self.returns_history) < 2:
            return 0.0
        
        # Use recent returns window
        recent_returns = self.returns_history[-self.reward_window:]
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            # No risk, return mean
            return mean_return * 10 if mean_return > 0 else 0.0
        
        # Sharpe Ratio (annualized approximation)
        sharpe = (mean_return / std_return) * np.sqrt(252)  # 252 trading days
        
        # Scale and clip
        reward = np.clip(sharpe, -5, 5)
        
        return float(reward)

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
        sharpe = self._calculate_sharpe_reward() if len(self.returns_history) > 1 else 0
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}, Sharpe: {sharpe:.3f}")
