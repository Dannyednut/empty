"""
Advanced PPO Trading Environment v3
Features:
- Continuous action space (position sizing)
- Stop-loss and take-profit
- Risk-adjusted rewards (Sharpe ratio)
- Realistic transaction costs
- Drawdown circuit breaker
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment with risk management
    
    Action Space: Continuous [-1, 1]
        -1.0: Full short (100% short position)
         0.0: Flat (no position)
        +1.0: Full long (100% long position)
    
    Observation Space: Box with shape (n_features + 3,)
        - All technical indicators
        - Current position (-1 to 1)
        - Unrealized PnL (%)
        - Current drawdown (%)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_cash: float = 10_000,
        episode_length: int = 2000,
        transaction_cost: float = 0.0002,  # 0.02% per trade (realistic for gold)
        max_position_size: float = 1.0,  # Maximum 100% of capital
        stop_loss_pct: float = 0.005,  # 0.5% stop loss (TIGHTER)
        take_profit_pct: float = 0.015,  # 1.5% take profit (REALISTIC)
        max_drawdown_limit: float = 0.20,  # 20% max drawdown circuit breaker
        reward_scaling: float = 1.0,
        lookback_window: int = 50,
        holding_penalty: float = 0.00001,  # Penalty per step held to force conviction
    ):
        super().__init__()
        
        self.df = df
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_scaling = reward_scaling
        self.lookback_window = lookback_window
        self.holding_penalty = holding_penalty
        
        # Action space: continuous position sizing
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: features + position + unrealized_pnl + drawdown
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 3,), dtype=np.float32
        )
        
        # Episode tracking
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Randomized episode start for better generalization
        max_start = len(self.df) - self.episode_length
        if max_start <= 0:
            self.start_idx = 0
            self.episode_length = len(self.df) - 1
        else:
            self.start_idx = np.random.randint(0, max_start)
        
        self.step_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        
        # Portfolio state
        self.cash = self.initial_cash
        self.position_units = 0.0  # Number of shares/units held
        self.last_action = 0.0  # Last position size requested (-1 to 1)
        self.entry_price = 0.0  # Average entry price
        
        # Performance tracking
        self.portfolio_values = [self.initial_cash]
        self.returns = []
        self.trades = []
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0
        
        # Episode metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get technical features
        row = self.df.iloc[self.step_idx]
        features = row[self.feature_columns].values.astype(np.float32)
        
        # Add portfolio state
        current_price = row['close']
        portfolio_value = self.cash + self.position_units * current_price
        
        # Unrealized PnL (%)
        if self.position_units != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * np.sign(self.position_units)
        else:
            unrealized_pnl = 0.0
        
        # Current drawdown (%)
        drawdown = self.current_drawdown
        
        # Combine all observations
        obs = np.concatenate([
            features,
            [self.last_action],  # Current action-based position (-1 to 1)
            [unrealized_pnl],  # Unrealized PnL
            [drawdown]  # Current drawdown
        ]).astype(np.float32)
        
        # CRITICAL: Replace any NaN values with 0.0 to prevent training crashes
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def _execute_trade(self, target_position: float, current_price: float) -> float:
        """
        Execute trade to reach target position using net accounting
        target_position: float from -1.0 to 1.0
        Returns: transaction cost incurred
        """
        # 1. Calculate current equity (Net Liquidation Value)
        portfolio_value = self.cash + (self.position_units * current_price)
        
        # 2. Limit portfolio value to prevent extreme numbers
        portfolio_value = np.clip(portfolio_value, 0, 1e12)
        
        # 3. Calculate target units based on target position
        # target_position of 1.0 means 100% of equity is long
        target_notional = portfolio_value * target_position
        target_units = target_notional / (current_price + 1e-8)
        
        # 4. Calculate change and cost
        units_to_trade = target_units - self.position_units
        
        if abs(units_to_trade) < 1e-8:
            return 0.0
            
        trade_notional = abs(units_to_trade) * current_price
        cost = trade_notional * self.transaction_cost
        
        # 5. Update state
        # New Cash = Old Cash - (Cost of buying new units) - Transaction Cost
        self.cash -= (units_to_trade * current_price) + cost
        
        # Win/Loss tracking for metrics
        if self.position_units != 0:
            pnl_pct = (current_price - self.entry_price) / (self.entry_price + 1e-8) * np.sign(self.position_units)
            if np.sign(target_units) != np.sign(self.position_units) or target_units == 0:
                # Closing or reversing position
                self.trades.append({'pnl': pnl_pct})
                if pnl_pct > 0: self.winning_trades += 1
                self.total_pnl += pnl_pct
                self.total_trades += 1
        
        if self.position_units == 0 or np.sign(target_units) != np.sign(self.position_units):
            self.entry_price = current_price
            
        self.position_units = target_units
        self.last_action = target_position
            
        # 6. Safety clip
        self.cash = np.clip(self.cash, -1e12, 1e12)
        
        return cost
    
    def _check_stop_loss_take_profit(self, current_price: float) -> bool:
        """
        Check if stop-loss or take-profit is hit
        Returns: True if position should be closed
        """
        if self.position_units == 0 or self.entry_price == 0:
            return False
        
        pnl_pct = (current_price - self.entry_price) / self.entry_price * np.sign(self.position_units)
        
        # Stop loss hit
        if pnl_pct <= -self.stop_loss_pct:
            return True
        
        # Take profit hit
        if pnl_pct >= self.take_profit_pct:
            return True
        
        return False
    
    def _calculate_reward(self, prev_value: float, current_value: float) -> float:
        """
        Calculate risk-adjusted reward with selective conviction
        """
        # 1. Basic return - normalized to portfolio value
        step_return = (current_value - prev_value) / (prev_value + 1e-8)
        step_return = np.clip(step_return, -0.1, 0.1)
        
        self.returns.append(step_return)
        reward = step_return
        
        # 2. Holding penalty (discourage aimless position holding)
        if self.position_units != 0:
            reward -= self.holding_penalty
        
        # 3. Sharpe ratio bonus (conviction consistency)
        if len(self.returns) >= self.lookback_window:
            recent_returns = self.returns[-self.lookback_window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8
            sharpe = mean_return / std_return
            reward += np.clip(sharpe * 0.005, -0.005, 0.005)
        
        # 4. Drawdown penalty
        if self.current_drawdown < -0.05:  # Start penalizing at 5% drawdown
            penalty = np.clip(abs(self.current_drawdown) * 0.2, 0, 0.1)
            reward -= penalty
        
        # 5. Final safety clip
        reward = np.nan_to_num(reward, nan=0.0)
        reward = np.clip(reward, -10, 10)
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        # Clip action to valid range
        target_position = np.clip(action[0], -self.max_position_size, self.max_position_size)
        
        # Get current price
        current_price = self.df.iloc[self.step_idx]['close']
        prev_value = self.cash + self.position_units * current_price
        
        # Check stop-loss / take-profit
        if self._check_stop_loss_take_profit(current_price):
            target_position = 0.0  # Force close position
        
        # Execute trade
        transaction_cost = self._execute_trade(target_position, current_price)
        
        # Move to next step
        self.step_idx += 1
        done = self.step_idx >= self.end_idx
        
        # Calculate new portfolio value
        if not done:
            next_price = self.df.iloc[self.step_idx]['close']
        else:
            next_price = current_price
        
        current_value = self.cash + self.position_units * next_price
        self.portfolio_values.append(current_value)
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Safe drawdown calculation with clipping
        if self.peak_value > 0:
            self.current_drawdown = (current_value - self.peak_value) / self.peak_value
            self.current_drawdown = np.clip(self.current_drawdown, -1.0, 0.0)  # Drawdown is always negative
        else:
            self.current_drawdown = 0.0
        
        # Check max drawdown circuit breaker
        if self.current_drawdown < -self.max_drawdown_limit:
            done = True
            current_value = self.peak_value * (1 - self.max_drawdown_limit)  # Cap loss
        
        # Calculate reward
        reward = self._calculate_reward(prev_value, current_value)
        
        # Get next observation
        obs = self._get_observation() if not done else self._get_observation()
        
        # Info dict
        info = {
            'portfolio_value': current_value,
            'position': self.last_action,
            'drawdown': self.current_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        }
        
        return obs, reward, done, False, info
    
    def render(self, mode='human'):
        """Render environment state"""
        current_price = self.df.iloc[self.step_idx]['close']
        portfolio_value = self.cash + self.position_units * current_price
        
        print(f"Step: {self.step_idx - self.start_idx}/{self.episode_length}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Position: {self.last_action:.2%}")
        print(f"Drawdown: {self.current_drawdown:.2%}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.winning_trades / self.total_trades if self.total_trades > 0 else 0:.2%}")
        print("-" * 50)
    
    def get_episode_metrics(self) -> dict:
        """Get comprehensive episode metrics"""
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Calculate Sharpe ratio
        if len(self.returns) > 1:
            sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-8) * np.sqrt(252 * 24 * 12)  # Annualized for 5min data
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        peak = self.initial_cash
        max_dd = 0.0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Win rate and profit factor
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        winning_pnl = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        losing_pnl = abs(sum([t['pnl'] for t in self.trades if t['pnl'] < 0]))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0.0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.total_pnl
        }


if __name__ == "__main__":
    # Test the environment
    print("Testing Advanced Trading Environment...")
    
    # Load data
    import sys
    sys.path.append('.')
    from features.indicators_v2 import build_features, get_feature_columns
    
    df = pd.read_csv("data/xauusd_m5.csv")
    df = build_features(df, base_timeframe='5min')
    feature_cols = get_feature_columns()
    
    # Create environment
    env = AdvancedTradingEnv(
        df=df,
        feature_columns=feature_cols,
        initial_cash=10_000,
        episode_length=1000
    )
    
    # Test random actions
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    done = False
    steps = 0
    while not done and steps < 100:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if steps % 20 == 0:
            env.render()
    
    # Get episode metrics
    metrics = env.get_episode_metrics()
    print("\nEpisode Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
