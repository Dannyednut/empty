import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PPOTradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10_000, episode_length=2000):
        super().__init__()

        self.df = df
        self.initial_cash = initial_cash
        self.episode_length = episode_length

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        # Randomized episode start
        max_start = len(self.df) - self.episode_length
        if max_start <= 0:
            self.start_idx = 0
            self.episode_length = len(self.df) - 1
        else:
            self.start_idx = np.random.randint(0, max_start)
        self.step_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length

        self.cash = self.initial_cash
        self.position = 0
        self.shares = 0
        self.entry_price = 0.0  # track buy price for exit reward

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
        reward = 0.0

        # Mask invalid actions
        if action == 1 and self.position == 1:
            action = 0  # already long, cannot buy
        if action == 2 and self.position == 0:
            action = 0  # flat, cannot sell

        # Execute actions
        if action == 1 and self.position == 0:  # Buy
            self.shares = self.cash / price
            self.cash = 0
            self.position = 1
            self.entry_price = price

        elif action == 2 and self.position == 1:  # Sell
            trade_pnl = (price - self.entry_price) / self.entry_price
            reward += trade_pnl  # scaled exit reward
            self.cash = self.shares * price
            self.shares = 0
            self.position = 0
            self.entry_price = 0

        # Move to next step
        self.step_idx += 1
        done = self.step_idx >= self.end_idx

        # Calculate portfolio value change
        next_price = self.df.iloc[self.step_idx]["close"]
        current_value = self.cash + self.shares * next_price
        step_reward = (current_value - prev_value) / self.initial_cash

        # Total reward: normalized PnL + exit reward - small overtrading penalty
        reward += step_reward
        reward -= 0.0002 * abs(action - 1)  # small penalty to discourage unnecessary trades

        return self._get_state(), reward, done, False, {}
