"""
PPO Trading Environment v6 (Apex Survivalist)
Features:
- Survivalist Reward Multiplier (Negative time-decay for profit stagnation)
- Divergence & Impulse Synergy (Apex Features)
- Momentum Acceleration Observation
- Profit Shielding (Early exit bonus)
- High-Frequency Polling Compatible
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class SniperApexEnvV6(gym.Env):
    """
    "Apex" Environment - Focused on Reversal Prediction and Survival.
    Teaches the model to be 'impatient' with stale profits.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_cash: float = 10_000,
        episode_length: int = 2880,       # ~2 days of M1 data for better regime context
        transaction_cost: float = 0.00010, # More aggressive spread for competitive training
        max_drawdown_limit: float = 0.08,  # Extra tight for "Apex" survival
    ):
        super().__init__()
        
        self.df = df
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        
        # ACTION SPACE: [Position (-1 to 1), ATR Multiplier (0.5 to 3.0), Flush Trigger (0 to 1)]
        # Flush Trigger > 0.8 forces an immediate exit regardless of pos conviction
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5, 0.0]), 
            high=np.array([1.0, 3.0, 1.0]), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: Features + [Pos, Mult, PnL, DD, Stagnation, Pulse]
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 6,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Curriculum: Allow starting anywhere to see random market regimes
        max_start = len(self.df) - self.episode_length - 100
        self.start_idx = np.random.randint(50, max_start) if max_start > 50 else 0
        
        self.step_idx = self.start_idx
        self.pulse_phase = 0
        
        self.cash = self.initial_cash
        self.position_units = 0.0
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.last_action = np.zeros(3)
        self.trade_duration = 0 # Track bars in trade
        self.peak_unrealized_pnl = 0.0
        self.stagnation_counter = 0.0
        
        self.portfolio_values = [self.initial_cash]
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0
        
        return self._get_observation(), {}

    def _get_pulse_price(self) -> float:
        row = self.df.iloc[self.step_idx]
        if self.pulse_phase == 0: return (row['open'] + row['high']) / 2
        elif self.pulse_phase == 1: return (row['open'] + row['low']) / 2
        else: return row['close']

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
            [self.last_action[0]], # Position
            [self.last_action[1]], # Multiplier
            [pnl],
            [self.current_drawdown],
            [self.stagnation_counter / 50.0], # Awareness of 'boring' trades
            [self.pulse_phase / 2.0]
        ]).astype(np.float32)
        
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        target_pos = np.clip(action[0], -1.0, 1.0)
        atr_mult = np.clip(action[1], 0.5, 3.0)
        flush_cmd = action[2]
        
        prev_price = self._get_pulse_price()
        prev_value = self.cash + (self.position_units * prev_price)
        
        # 1. Survival Check: Early Exit (Flush)
        emergency_exit = (flush_cmd > 0.8 and self.position_units != 0)
        
        atr_val = self.df.iloc[self.step_idx].get('atr', prev_price * 0.001)
        
        # 2. Reversal Prediction logic (Force flip if target_pos flips sign)
        opening = (self.position_units == 0 and abs(target_pos) > 0.1)
        reversing = (np.sign(target_pos) != np.sign(self.position_units) and self.position_units != 0)
        
        # Cache direction for reward logic before potential reset
        self.pre_reset_pos_sign = np.sign(self.position_units) if self.position_units != 0 else 0
        
        if opening or reversing:
            self.entry_price = prev_price if prev_price > 0 else 1.0
            dist = atr_val * atr_mult
            self.current_sl = self.entry_price - (dist if target_pos > 0 else -dist)
            self.current_tp = self.entry_price + (dist * 2.8 if target_pos > 0 else -dist * 2.8)
            self.trade_duration = 0
            self.stagnation_counter = 0
            self.peak_unrealized_pnl = 0

        # Physical Execution
        if emergency_exit:
            self.cash += (self.position_units * prev_price) - (abs(self.position_units) * prev_price * self.transaction_cost)
            self.position_units = 0
            target_pos = 0 # Override
        
        target_units = (prev_value * target_pos) / (prev_price + 1e-9)
        units_to_trade = target_units - self.position_units
        cost = abs(units_to_trade) * prev_price * self.transaction_cost
        
        self.cash -= (units_to_trade * prev_price) + cost
        self.position_units = target_units
        self.last_action = action
        
        # Time Advance
        self.pulse_phase += 1
        if self.pulse_phase > 2:
            self.pulse_phase = 0
            self.step_idx += 1
            if self.position_units != 0: self.trade_duration += 1
            
        done = self.step_idx >= (self.start_idx + self.episode_length)
        curr_price = self._get_pulse_price()
        
        # SL/TP hit logic (Dynamic Reversal Catching)
        if self.position_units != 0:
            hit = False
            if self.position_units > 0:
                if curr_price <= self.current_sl or curr_price >= self.current_tp: hit = True
            else:
                if curr_price >= self.current_sl or curr_price <= self.current_tp: hit = True
            
            if hit:
                exit_p = self.current_sl if ((self.position_units > 0 and curr_price <= self.current_sl) or 
                                           (self.position_units < 0 and curr_price >= self.current_sl)) else self.current_tp
                self.cash += (self.position_units * exit_p) - (abs(self.position_units) * exit_p * self.transaction_cost)
                self.position_units = 0

        # 3. APEX REWARD LOGIC
        curr_value = self.cash + (self.position_units * curr_price)
        step_return = (curr_value - prev_value) / (prev_value + 1e-9)
        reward = step_return * 50.0 
        
        # Calculate u_pnl for reward context (even if just closed)
        ep = self.entry_price if self.entry_price > 0 else 1.0
        # If we still have units, use current price. If we just closed, use price of closing (curr or exit_p logic)
        u_pnl = (curr_price - ep) / ep * np.sign(self.position_units) if self.position_units != 0 else 0.0

        if self.position_units != 0:
            # --- IMPATIENT REAPER MECHANISM ---
            if u_pnl > self.peak_unrealized_pnl:
                self.peak_unrealized_pnl = u_pnl
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
            # Penalize sitting in a winning trade that stopped moving
            if u_pnl > 0.001 and self.stagnation_counter > 10:
                reward -= (u_pnl * (self.stagnation_counter / 1000.0))
            
            # Reversal Penalty: If PnL starts dropping from peak, punish hard
            drawdown_from_peak_pnl = self.peak_unrealized_pnl - u_pnl
            if drawdown_from_peak_pnl > 0.0005:
                reward -= drawdown_from_peak_pnl * 100.0 
        
        # Survival/Panic Evaluation (If we emergency exited this step)
        if emergency_exit:
            # Use cached sign to see if we closed a winner or loser
            flush_pnl = (prev_price - ep) / ep * self.pre_reset_pos_sign
            if flush_pnl > 0.0002:
                reward += flush_pnl * 10.0 # Reward for protection
            elif flush_pnl < 0:
                reward -= 0.1 # Penalty for panic

        self.portfolio_values.append(curr_value)
        if curr_value > self.peak_value: self.peak_value = curr_value
        self.current_drawdown = (curr_value - self.peak_value) / (self.peak_value + 1e-9)
        
        if self.current_drawdown < -self.max_drawdown_limit:
            done = True
            reward -= 2.0

        info = {
            'portfolio_value': curr_value,
            'drawdown': self.current_drawdown,
            'stagnation': self.stagnation_counter
        }
            
        return self._get_observation(), reward, done, False, info
