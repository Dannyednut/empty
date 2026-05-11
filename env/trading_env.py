import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class AdvancedTradingEnv(gym.Env):
    """
    Advanced PPO Trading Environment v4
    Pro-grade, defensive trader environment
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_cash: float = 10_000,
        episode_length: int = 2000,
        transaction_cost: float = 0.0002,
        max_risk_per_trade: float = 0.01,   # 1% hard cap
        max_drawdown_limit: float = 0.20,
        min_trade_gap: int = 5
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_risk = max_risk_per_trade
        self.max_drawdown_limit = max_drawdown_limit
        self.min_trade_gap = min_trade_gap

        # --- Action space ---
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([ 1, 1, 1, 1, 1], dtype=np.float32),
        )

        # --- Observation space ---
        n_features = len(feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 7,),
            dtype=np.float32
        )

        self.reset()

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        max_start = len(self.df) - self.episode_length
        self.start_idx = np.random.randint(0, max(1, max_start))
        self.step_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length

        # Portfolio
        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0

        # Trade state
        self.position = 0.0
        self.entry_price = None
        self.stop_price = None
        self.take_price = None
        self.time_in_trade = 0
        self.last_trade_step = -self.min_trade_gap

        # Tracking
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.returns = []

        return self._get_observation(), {}

    # ==========================================================
    # Observation
    # ==========================================================

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        features = row[self.feature_columns].values.astype(np.float32)

        unrealized_pnl = 0.0
        dist_stop = 0.0
        dist_take = 0.0

        if self.position != 0:
            price = row['close']
            unrealized_pnl = (
                (price - self.entry_price)
                / (self.entry_price + 1e-8)
                * np.sign(self.position)
            )
            dist_stop = abs(price - self.stop_price)
            dist_take = abs(self.take_price - price)

        obs = np.concatenate([
            features,
            [
                self.position,
                unrealized_pnl,
                self.current_drawdown,
                self.time_in_trade,
                dist_stop,
                dist_take,
                self.equity / self.initial_cash - 1
            ]
        ])

        return np.nan_to_num(obs, 0.0)

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, action: np.ndarray):
        # --- Action smoothing ---
        action = 0.85 * self.prev_action + 0.15 * action
        self.prev_action = action.copy()

        direction, risk_frac, stop_k, take_k, exit_sig = action
        row = self.df.iloc[self.step_idx]
        price = row['close']
        atr = row.get('ATR', price * 0.001)

        reward = 0.0
        done = False

        prev_equity = self.equity

        # =========================
        # Exit logic
        # =========================
        if self.position != 0:
            self.time_in_trade += 1

            stop_hit = price <= self.stop_price if self.position > 0 else price >= self.stop_price
            take_hit = price >= self.take_price if self.position > 0 else price <= self.take_price

            if stop_hit or take_hit or exit_sig > 0.7:
                self._close_trade(price)

        # =========================
        # Entry logic
        # =========================
        can_trade = (self.step_idx - self.last_trade_step) >= self.min_trade_gap

        if self.position == 0 and can_trade:
            if abs(direction) > 0.6 and risk_frac > 0.1:
                self._open_trade(direction, risk_frac, stop_k, take_k, price, atr)

        # =========================
        # Advance time
        # =========================
        self.step_idx += 1
        done |= self.step_idx >= self.end_idx

        # =========================
        # Equity & drawdown
        # =========================
        self.peak_value = max(self.peak_value, self.equity)
        self.current_drawdown = (self.equity - self.peak_value) / self.peak_value

        if self.current_drawdown < -self.max_drawdown_limit:
            done = True
            reward -= 1.0  # catastrophic penalty

        # =========================
        # Reward (minimal, defensive)
        # =========================
        step_return = (self.equity - prev_equity) / (prev_equity + 1e-8)
        reward += np.clip(step_return, -0.05, 0.05)

        self.returns.append(step_return)

        return self._get_observation(), reward, done, False, {}

    # ==========================================================
    # Trade helpers
    # ==========================================================

    def _open_trade(self, direction, risk_frac, stop_k, take_k, price, atr):
        risk = self.max_risk * np.clip(risk_frac, 0, 1)
        self.position = np.sign(direction) * risk

        self.entry_price = price
        self.stop_price = price - np.sign(direction) * (0.5 + stop_k) * atr
        self.take_price = price + np.sign(direction) * (1.0 + take_k) * atr

        self.time_in_trade = 0
        self.last_trade_step = self.step_idx

    def _close_trade(self, price):
        pnl = self.position * (price - self.entry_price)
        pnl -= abs(pnl) * self.transaction_cost

        self.equity += pnl

        self.position = 0.0
        self.entry_price = None
        self.stop_price = None
        self.take_price = None
        self.time_in_trade = 0

    # ==========================================================
    # Render
    # ==========================================================

    def render(self, mode='human'):
        print(f"Step {self.step_idx}")
        print(f"Equity: {self.equity:.2f}")
        print(f"Position: {self.position}")
        print(f"Drawdown: {self.current_drawdown:.2%}")
        print("-" * 40)
