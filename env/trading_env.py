import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PPOTradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10_000):
        super().__init__()

        self.df = df
        self.initial_cash = initial_cash

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.position = 0
        self.shares = 0
        return self._get_state(), {}

    def _get_state(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row["diff"],
            row["diff_prev"],
            row["candle_range"],
            self.position
        ], dtype=np.float32)

    def step(self, action):
        price = self.df.iloc[self.step_idx]["close"]
        prev_value = self.cash + self.shares * price

        # Actions
        if action == 1 and self.position == 0:
            self.shares = self.cash / price
            self.cash = 0
            self.position = 1

        elif action == 2 and self.position == 1:
            self.cash = self.shares * price
            self.shares = 0
            self.position = 0

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1

        next_price = self.df.iloc[self.step_idx]["close"]
        current_value = self.cash + self.shares * next_price

        reward = current_value - prev_value
        reward -= 0.0002 * abs(action - 1)  # overtrading penalty

        return self._get_state(), reward, done, False, {}
