"""
PPO Trading Environment v5 (Pulse Engine)
Features:
- Intra-Bar Pulse Processing (Sub-candle resolution)
- 3-Stage Price Synthesis (Training on OHLC sub-components)
- Momentum-Pulse Rewards (Speed of Profit)
- High-Frequency Polling Compatible (1s - 5s ready)
- Temporal Feature Injection (Time-to-Candidate-Close)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class SniperPulseEnvV5(gym.Env):
    """
    High-Frequency "Pulse" Environment.
    Each 1-minute candle is split into 3 synthetic "Pulses" to simulate intra-bar movement.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_cash: float = 10_000,
        episode_length: int = 1440, # ~1 day of M1 data
        transaction_cost: float = 0.00012, # Institutional tight spread
        max_drawdown_limit: float = 0.12,  # Tighter risk for high-freq
    ):
        super().__init__()
        
        self.df = df
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        
        # ACTION SPACE: [Position (-1 to 1), ATR Multiplier (0.5 to 3.0)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5]), 
            high=np.array([1.0, 3.0]), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: Features + [Pos, Mult, PnL, DD, Pulse_State]
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 5,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Ensure we have enough data for lookback and episode
        max_start = len(self.df) - self.episode_length - 100
        self.start_idx = np.random.randint(50, max_start) if max_start > 50 else 0
        
        self.step_idx = self.start_idx
        self.pulse_phase = 0 # 0: OH/2, 1: OL/2, 2: Close
        
        self.cash = self.initial_cash
        self.position_units = 0.0
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.last_action = np.zeros(2)
        
        self.portfolio_values = [self.initial_cash]
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0
        
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}

    def _get_pulse_price(self) -> float:
        """Synthesize intra-candle price based on phase"""
        row = self.df.iloc[self.step_idx]
        if self.pulse_phase == 0:
            return (row['open'] + row['high']) / 2
        elif self.pulse_phase == 1:
            return (row['open'] + row['low']) / 2
        else:
            return row['close']

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        features = row[self.feature_columns].values.astype(np.float32)
        
        cur_price = self._get_pulse_price()
        pnl = 0.0
        if self.position_units != 0:
            ep = self.entry_price if self.entry_price > 0 else 1.0
            pnl = (cur_price - ep) / ep * np.sign(self.position_units)
            
        obs = np.concatenate([
            features,
            [self.last_action[0]], 
            [self.last_action[1]], 
            [pnl],
            [self.current_drawdown],
            [self.pulse_phase / 2.0] # State of the current candle
        ]).astype(np.float32)
        
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        target_pos = np.clip(action[0], -1.0, 1.0)
        atr_mult = np.clip(action[1], 0.5, 3.0)
        
        prev_price = self._get_pulse_price()
        prev_value = self.cash + (self.position_units * prev_price)
        
        # ATR Scale Logic
        atr_val = self.df.iloc[self.step_idx].get('atr', prev_price * 0.001)
        
        # Handle Trade Logic (Entry Price & Stop Alignment)
        opening = (self.position_units == 0 and abs(target_pos) > 0.01) # Low threshold for any move
        reversing = (np.sign(target_pos) != np.sign(self.position_units) and self.position_units != 0 and abs(target_pos) > 0.01)
        
        if opening or reversing:
            self.entry_price = prev_price if prev_price > 0 else 1.0 # Safety
            dist = atr_val * atr_mult
            if target_pos > 0:
                self.current_sl = self.entry_price - dist
                self.current_tp = self.entry_price + (dist * 2.8) 
            else:
                self.current_sl = self.entry_price + dist
                self.current_tp = self.entry_price - (dist * 2.8)
            self.total_trades += 1

        # Execution
        target_units = (prev_value * target_pos) / (prev_price + 1e-9)
        units_to_trade = target_units - self.position_units
        cost = abs(units_to_trade) * prev_price * self.transaction_cost
        
        self.cash -= (units_to_trade * prev_price) + cost
        self.position_units = target_units
        self.last_action = action
        
        # Pulse Advance Logic
        self.pulse_phase += 1
        if self.pulse_phase > 2:
            self.pulse_phase = 0
            self.step_idx += 1
            
        done = self.step_idx >= (self.start_idx + self.episode_length)
        curr_price = self._get_pulse_price()
        
        # Dynamic Pulse Exit
        if self.position_units != 0:
            hit = False
            if self.position_units > 0:
                if curr_price <= self.current_sl or curr_price >= self.current_tp: hit = True
            else:
                if curr_price >= self.current_sl or curr_price <= self.current_tp: hit = True
                
            if hit:
                exit_p = self.current_sl if ((self.position_units > 0 and curr_price <= self.current_sl) or 
                                           (self.position_units < 0 and curr_price >= self.current_sl)) else self.current_tp
                ep = self.entry_price if self.entry_price > 0 else 1.0
                pnl_p = (exit_p - ep) / ep * np.sign(self.position_units)
                if pnl_p > 0: self.winning_trades += 1
                
                self.cash += (self.position_units * exit_p) - (abs(self.position_units) * exit_p * self.transaction_cost)
                self.position_units = 0
                self.last_action[0] = 0

        # Reward Pulse
        curr_value = self.cash + (self.position_units * curr_price)
        step_return = (curr_value - prev_value) / (prev_value + 1e-9)
        
        reward = step_return * 20.0 # High sensitivity
        
        # Pulse Efficiency Bonus
        if self.position_units != 0:
            # Reward for unrealized PnL growing quickly
            ep = self.entry_price if self.entry_price > 0 else 1.0
            mom = (curr_price - ep) / ep * np.sign(self.position_units)
            reward += mom * 5.0 
            
            # Decay penalty: Scalpers shouldn't sit in trades
            reward -= 0.0001 

        if step_return < 0:
            reward *= 1.5 # Harsh Sortino penalty
            
        self.portfolio_values.append(curr_value)
        if curr_value > self.peak_value: self.peak_value = curr_value
        self.current_drawdown = (curr_value - self.peak_value) / (self.peak_value + 1e-9)
        
        if self.current_drawdown < -self.max_drawdown_limit:
            done = True
            reward -= 1.0

        info = {
            'portfolio_value': curr_value,
            'drawdown': self.current_drawdown,
            'pulse': self.pulse_phase
        }
            
        return self._get_observation(), reward, done, False, info
