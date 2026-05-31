import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class TradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df: pd.DataFrame, 
                 llm_embeddings: Optional[np.ndarray] = None,
                 window_size: int = 30,
                 initial_balance: float = 100000.0, 
                 transaction_fee: float = 0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.llm_embeddings = llm_embeddings
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Dict
        # 'market': (window_size, input_dim)
        # 'news': (embedding_dim,)
        # 'account': (2,)
        
        self.input_dim = len(df.columns) 
        self.llm_dim = llm_embeddings.shape[1] if llm_embeddings is not None else 768
        
        self.observation_space = spaces.Dict({
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, self.input_dim), dtype=np.float32),
            'news': spaces.Box(low=-np.inf, high=np.inf, shape=(self.llm_dim,), dtype=np.float32),
            'account': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        
        # State variables
        self.balance = initial_balance
        self.holdings = 0
        self.current_step = window_size 
        self.max_steps = len(df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = self.window_size
        
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute Action
        if action == 1: # Buy
            max_shares = self.balance // (current_price * (1 + self.transaction_fee))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.holdings += max_shares
        elif action == 2: # Sell
            if self.holdings > 0:
                revenue = self.holdings * current_price * (1 - self.transaction_fee)
                self.balance += revenue
                self.holdings = 0
        
        # Step forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Reward
        new_price = self.df.iloc[self.current_step]['Close'] if not terminated else current_price
        portfolio_value = self.balance + (self.holdings * new_price)
        reward = portfolio_value - self.initial_balance 
        
        obs = self._get_observation()
        info = {'portfolio_value': portfolio_value}
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        # 1. Market Window
        window_data = self.df.iloc[self.current_step - self.window_size : self.current_step].values.astype(np.float32)
        
        # 2. LLM Embedding
        if self.llm_embeddings is not None:
            llm_vec = self.llm_embeddings[self.current_step].astype(np.float32)
        else:
            llm_vec = np.zeros(self.llm_dim, dtype=np.float32)
            
        # 3. Account State
        account_vec = np.array([self.balance, self.holdings], dtype=np.float32)
        
        return {
            'market': window_data,
            'news': llm_vec,
            'account': account_vec
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}")

if __name__ == "__main__":
    # Test
    # Create dummy dataframe
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {'Close': np.random.uniform(100, 200, size=100)}
    df = pd.DataFrame(data, index=dates)
    
    env = TradingEnv(df)
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        env.render()
