import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import PPOTradingEnv
from features.indicators import build_features

df = pd.read_csv("data/xauusd_m5.csv")
df = build_features(df)

test_df = df.iloc[int(len(df) * 0.7):]

env = PPOTradingEnv(test_df)
model = PPO.load("xauusd_ppo_trader")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

final_value = env.cash + env.shares * test_df.iloc[env.step_idx]["close"]
print("Final portfolio value:", final_value)
