"""
PPO Trading Environment v4 (The Sniper)
Features:
- Multi-dimensional action space: [Position, ATR_Multiplier]
- Dynamic Volatility-Adjusted Stop Loss/Take Profit
- Potential-based Rewards (Entry Timing Alpha)
- Sortino-Ratio Utility Function
- Spread and Liquidity Sensitivity
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SniperTradingEnv(gym.Env):
    """
    Institutional-grade Sniper environment for XAU/USD.
    Teaches directional alpha AND entry timing through ATR-scaled logic.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_cash: float = 10_000,
        episode_length: int = 2000,
        transaction_cost: float = 0.00015,  # Tighter institutional spread
        max_drawdown_limit: float = 0.15,   # Focused risk
        reward_scaling: float = 1.0,
        lookback_window: int = 50,
    ):
        super().__init__()
        
        self.df = df
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_scaling = reward_scaling
        self.lookback_window = lookback_window
        
        # ACTION SPACE: Multi-Discrete or Multi-Continuous
        # 0: Position (-1 to 1) -> Direction & Size
        # 1: ATR Multiplier (0.5 to 3.0) -> How tight is the stop?
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5]), 
            high=np.array([1.0, 3.0]), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: Features + State
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        max_start = len(self.df) - self.episode_length - 50
        self.start_idx = np.random.randint(50, max_start) if max_start > 50 else 0
        
        self.step_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        
        self.cash = self.initial_cash
        self.position_units = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.last_action = np.zeros(2)
        
        self.portfolio_values = [self.initial_cash]
        self.returns = []
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0
        
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        features = row[self.feature_columns].values.astype(np.float32)
        
        # Net state
        cur_price = row['close']
        pnl = 0.0
        if self.position_units != 0:
            pnl = (cur_price - self.entry_price) / self.entry_price * np.sign(self.position_units)
            
        obs = np.concatenate([
            features,
            [self.last_action[0]], # Prev Pos
            [self.last_action[1]], # Prev SL Mult
            [pnl],
            [self.current_drawdown]
        ]).astype(np.float32)
        
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Parse actions
        target_pos = np.clip(action[0], -1.0, 1.0)
        atr_mult = np.clip(action[1], 0.5, 3.0)
        
        prev_price = self.df.iloc[self.step_idx]['close']
        prev_value = self.cash + (self.position_units * prev_price)
        
        # 2. ATR Scale Logic (Volatility-Adjusted Stops)
        # Check if atr column exists, else fallback
        atr_val = self.df.iloc[self.step_idx].get('atr', prev_price * 0.002)
        
        # Handle trades
        closing = (target_pos == 0 and self.position_units != 0)
        reversing = (np.sign(target_pos) != np.sign(self.position_units) and self.position_units != 0)
        opening = (self.position_units == 0 and target_pos != 0)
        
        if opening or reversing:
            # Set Dynamic SL/TP based on ATR
            self.entry_price = prev_price
            self.entry_idx = self.step_idx
            dist = atr_val * atr_mult
            
            if target_pos > 0: # Long
                self.current_sl = self.entry_price - dist
                self.current_tp = self.entry_price + (dist * 2.5) # RR 1:2.5
            else: # Short
                self.current_sl = self.entry_price + dist
                self.current_tp = self.entry_price - (dist * 2.5)
            
            self.total_trades += 1

        # 3. Execution & Cost
        target_units = (prev_value * target_pos) / (prev_price + 1e-8)
        units_to_trade = target_units - self.position_units
        cost = abs(units_to_trade) * prev_price * self.transaction_cost
        
        self.cash -= (units_to_trade * prev_price) + cost
        self.position_units = target_units
        self.last_action = action
        
        # 4. Advance 
        self.step_idx += 1
        done = self.step_idx >= self.end_idx
        
        curr_price = self.df.iloc[self.step_idx]['close']
        
        # 5. Dynamic Exit Check
        if self.position_units != 0:
            hit = False
            if self.position_units > 0: # Long
                if curr_price <= self.current_sl or curr_price >= self.current_tp: hit = True
            else: # Short
                if curr_price >= self.current_sl or curr_price <= self.current_tp: hit = True
                
            if hit:
                # Close at SL/TP price
                exit_p = self.current_sl if (self.position_units > 0 and curr_price <= self.current_sl) else self.current_tp
                # Metric tracking
                pnl_p = (exit_p - self.entry_price) / self.entry_price * np.sign(self.position_units)
                if pnl_p > 0: self.winning_trades += 1
                
                # Execute close
                self.cash += (self.position_units * exit_p) - (abs(self.position_units) * exit_p * self.transaction_cost)
                self.position_units = 0
                self.last_action[0] = 0

        # 6. Reward Logic (The Sniper Engine)
        curr_value = self.cash + (self.position_units * curr_price)
        step_return = (curr_value - prev_value) / (prev_value + 1e-8)
        
        # --- ALPHA BONUSES ---
        reward = step_return * 10.0 # Scale return
        
        if opening or reversing:
            # Timing Penalty (Spread cost awareness)
            reward -= (cost / prev_value) * 1.5 
            
        # Entry Timing Reward (Momentum Potential)
        # Reward if the price moves favorably within 5 bars of entry
        if self.position_units != 0 and (self.step_idx - self.entry_idx) < 5:
            mom = (curr_price - self.entry_price) / self.entry_price * np.sign(self.position_units)
            if mom > 0:
                reward += mom * 2.0 # Extra bonus for early timing alpha
        
        # Sortino-Style Utility (Downside only penalty)
        if step_return < 0:
            reward *= 1.2 # Penalize losers harder than rewarding winners
            
        self.portfolio_values.append(curr_value)
        if curr_value > self.peak_value: self.peak_value = curr_value
        self.current_drawdown = (curr_value - self.peak_value) / (self.peak_value + 1e-8)
        
        if self.current_drawdown < -self.max_drawdown_limit:
            done = True
            reward -= 0.5 # Deep drawdown failure
            
        # Info dict
        info = {
            'portfolio_value': curr_value,
            'position': self.last_action[0],
            'drawdown': self.current_drawdown,
            'total_trades': self.total_trades
        }
            
        return self._get_observation(), reward, done, False, info

    def get_metrics(self) -> dict:
        total_ret = (self.portfolio_values[-1] - self.initial_cash) / self.initial_cash
        wr = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        return {
            'return': total_ret,
            'win_rate': wr,
            'trades': self.total_trades,
            'drawdown': self.current_drawdown
        }
