import pandas as pd
from stable_baselines3 import PPO
from env.trading_env_v2 import PPOTradingEnv
from features.indicators import build_features
from datetime import datetime

df = pd.read_csv("data/xauusd_m5.csv")
df = build_features(df)

# Train / test split
split = int(len(df) * 0.7)
train_df = df.iloc[:split]

env = PPOTradingEnv(train_df)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=300_000)
model.save(f"xauusd_m5_ppo_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
