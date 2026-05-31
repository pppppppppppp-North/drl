import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from src.models.feature_pipeline import FeaturePipeline

class TradingEnvSortino(gym.Env):
    """Trading environment with Sortino Ratio rewards (downside risk focus)."""
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 100000.0, 
        transaction_fee: float = 0.001,
        use_feature_pipeline: bool = True,
        sequence_length: int = 30,
        reward_window: int = 20,
        target_return: float = 0.0  # Minimum acceptable return (MAR)
    ):
        super(TradingEnvSortino, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.use_feature_pipeline = use_feature_pipeline
        self.reward_window = reward_window
        self.target_return = target_return
        
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
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
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
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        new_price = self.df.iloc[self.current_step]['Close'] if not terminated else current_price
        portfolio_value = self.balance + (self.holdings * new_price)
        
        # Calculate return
        step_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(step_return)
        self.prev_portfolio_value = portfolio_value
        
        # Sortino Ratio reward
        reward = self._calculate_sortino_reward()
        
        # Add trading bonus to encourage activity
        trading_bonus = 0.1 if action in [1, 2] else 0.0
        reward += trading_bonus
        
        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
            'sortino_reward': reward,
            'step_return': step_return
        }
        
        return obs, reward, terminated, truncated, info

    def _calculate_sortino_reward(self) -> float:
        """Calculate Sortino Ratio - only penalizes downside volatility."""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.reward_window:])
        
        mean_return = np.mean(recent_returns)
        
        # Downside deviation: only negative returns
        downside_returns = recent_returns[recent_returns < self.target_return]
        
        if len(downside_returns) == 0:
            # No downside risk - reward highly
            return mean_return * 20 if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return mean_return * 10 if mean_return > 0 else 0.0
        
        # Sortino Ratio (annualized)
        sortino = ((mean_return - self.target_return) / downside_std) * np.sqrt(252)
        
        # Scale and clip
        reward = np.clip(sortino, -5, 5)
        
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
        sortino = self._calculate_sortino_reward() if len(self.returns_history) > 1 else 0
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}, Sortino: {sortino:.3f}")
