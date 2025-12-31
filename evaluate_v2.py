import pandas as pd
from stable_baselines3 import PPO
from env.trading_env_v2 import PPOTradingEnv
from features.indicators import build_features

# --- Load and preprocess data ---
df = pd.read_csv("data/xauusd_m15.csv")
df = build_features(df)

# Use the last 30% of data as test set
test_df = df.iloc[int(len(df) * 0.7):].reset_index(drop=True)

# --- Initialize environment ---
env = PPOTradingEnv(test_df, episode_length=len(test_df))  # use full test period
model = PPO.load("xauusd_ppo_trader")

# --- Reset environment ---
obs, _ = env.reset()
done = False

portfolio_values = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    # Track portfolio value at each step
    current_price = test_df.iloc[min(env.step_idx, len(test_df)-1)]["close"]
    current_value = env.cash + env.shares * current_price
    portfolio_values.append(current_value)

# --- Final portfolio value ---
final_value = portfolio_values[-1]
print("Final portfolio value:", final_value)

# Optional: print PnL
pnl = final_value - env.initial_cash
print("Total PnL:", pnl)

# Optional: visualize equity curve
import matplotlib.pyplot as plt

plt.plot(portfolio_values)
plt.title("PPO Trader Equity Curve")
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.show()
