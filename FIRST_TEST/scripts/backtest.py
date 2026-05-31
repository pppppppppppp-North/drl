import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv
from src.data.loader import SET50Loader
from src.data.technicals import TechnicalIndicatorEngine
import argparse
import os

def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.0) -> dict:
    """
    Calculate performance metrics from returns.
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default 0.0)
    
    Returns:
        Dictionary of performance metrics
    """
    # Remove any NaN or infinite values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0
        }
    
    # Cumulative return
    total_return = (np.prod(1 + returns) - 1) * 100  # in percentage
    
    # Annualized metrics (assuming 252 trading days)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252) if std_return > 0 else 0.0
    
    # Sortino Ratio (only penalize downside volatility)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (mean_return - risk_free_rate/252) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    # Max Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100  # in percentage
    
    # Annualized Volatility
    volatility = std_return * np.sqrt(252) * 100  # in percentage
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility
    }

def backtest_agent(
    model_path: str,
    tickers: list,
    start_date: str,
    end_date: str,
    initial_balance: float = 100000.0,
    plot: bool = True
):
    """
    Backtest a trained DRL agent and compare with buy-and-hold.
    """
    print(f"=" * 70)
    print(f"BACKTESTING: {tickers} from {start_date} to {end_date}")
    print(f"=" * 70)
    
    # 1. Load Data
    loader = SET50Loader(tickers)
    try:
        df = loader.fetch_data(start_date, end_date)
        if df.empty:
            raise ValueError("No data fetched.")
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            ticker = df.columns.levels[0][0]
            df = df[ticker]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # 2. Add Technical Indicators
    ta_engine = TechnicalIndicatorEngine()
    df = ta_engine.add_indicators(df)
    df.dropna(inplace=True)
    
    print(f"Loaded {len(df)} data points.")
    
    # 3. Load Model
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}")
        return
    
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # 4. Create Environment and Run Agent
    env = TradingEnv(df, initial_balance=initial_balance)
    obs, info = env.reset()
    
    agent_portfolio_values = [initial_balance]
    actions_taken = []
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent_portfolio_values.append(info['portfolio_value'])
        actions_taken.append(action)
    
    # 5. Calculate Buy-and-Hold Baseline
    buy_hold_values = [initial_balance]
    shares_bought = initial_balance / df.iloc[0]['Close']
    
    for i in range(1, len(df)):
        buy_hold_values.append(shares_bought * df.iloc[i]['Close'])
    
    # 6. Compute Metrics
    agent_returns = np.diff(agent_portfolio_values) / agent_portfolio_values[:-1]
    buy_hold_returns = np.diff(buy_hold_values) / buy_hold_values[:-1]
    
    agent_metrics = calculate_metrics(agent_returns)
    buy_hold_metrics = calculate_metrics(buy_hold_returns)
    
    # 7. Print Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'DRL Agent':<20} {'Buy & Hold':<20}")
    print("-" * 70)
    print(f"{'Total Return (%)':<25} {agent_metrics['total_return']:>19.2f} {buy_hold_metrics['total_return']:>19.2f}")
    print(f"{'Sharpe Ratio':<25} {agent_metrics['sharpe_ratio']:>19.2f} {buy_hold_metrics['sharpe_ratio']:>19.2f}")
    print(f"{'Sortino Ratio':<25} {agent_metrics['sortino_ratio']:>19.2f} {buy_hold_metrics['sortino_ratio']:>19.2f}")
    print(f"{'Max Drawdown (%)':<25} {agent_metrics['max_drawdown']:>19.2f} {buy_hold_metrics['max_drawdown']:>19.2f}")
    print(f"{'Volatility (%)':<25} {agent_metrics['volatility']:>19.2f} {buy_hold_metrics['volatility']:>19.2f}")
    print(f"{'Final Portfolio ($)':<25} {agent_portfolio_values[-1]:>19.2f} {buy_hold_values[-1]:>19.2f}")
    print("=" * 70)
    
    # 8. Plot Results
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio Value Over Time
        ax1.plot(agent_portfolio_values, label='DRL Agent', linewidth=2)
        ax1.plot(buy_hold_values, label='Buy & Hold', linewidth=2, linestyle='--')
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Action Distribution
        action_labels = ['Hold', 'Buy', 'Sell']
        action_counts = [actions_taken.count(i) for i in range(3)]
        ax2.bar(action_labels, action_counts, color=['gray', 'green', 'red'])
        ax2.set_title('Action Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plot_path = 'results/backtest_results.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.show()
    
    return agent_metrics, buy_hold_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest a trained DRL trading agent')
    parser.add_argument('--model', type=str, default='checkpoints/ppo_trading_agent',
                        help='Path to trained model (without .zip extension)')
    parser.add_argument('--tickers', nargs='+', default=['PTT.BK'],
                        help='Stock tickers to backtest')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Backtest start date')
    parser.add_argument('--end', type=str, default='2024-01-01',
                        help='Backtest end date')
    parser.add_argument('--balance', type=float, default=100000.0,
                        help='Initial balance')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    backtest_agent(
        model_path=args.model,
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        plot=not args.no_plot
    )
