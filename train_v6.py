"""
PPO Sniper V6 - Apex Survivalist Trainer
Focused on Survival, Reversal Prediction, and Impatient Profit Taking.
"""
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.trading_env_v6_apex import SniperApexEnvV6
from features.indicators_v2 import build_features, get_feature_columns
from data.fetch_mt5 import fetch_data

def train_apex_sniper(symbol="xauusd", timeframe="m1", total_timesteps=2_000_000):
    print(f"🔥 Initializing Sniper Apex (V6) Survival Training for {symbol.upper()}...")
    
    # 1. Prepare High-Res Data (M1)
    data_path = f"data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        print(f"📡 Fetching fresh {timeframe.upper()} data...")
        df = fetch_data(symbol=symbol.upper(), timeframe=timeframe, bars=500_000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    print(f"📊 Building Apex Features (Divergence & Impulse)...")
    df = build_features(df, base_timeframe='1min')
    df = df.dropna()
    
    feature_columns = get_feature_columns()
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # 2. Setup Apex Environment
    # We increase episode length to see more regime shifts
    def make_env():
        return SniperApexEnvV6(
            df=df,
            feature_columns=feature_columns,
            episode_length=2880 # 2 days of M1
        )
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 3. Apex PPO Hyperparameters (Higher Entropy for Exploration of the Exit Door)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-5, # Ultra-conservative for precision
        n_steps=8192,      # Large buffer for survival context
        batch_size=1024,
        n_epochs=15,
        gamma=0.97,        # High focus on the "now" vs the far future
        gae_lambda=0.90,
        clip_range=0.1,
        ent_coef=0.03,      # Force the model to try different exit scenarios
        tensorboard_log="./logs/sniper_v6_apex/"
    )
    
    # 4. Mission Execution
    save_path = f"models/{symbol.lower()}/sniper_v6"
    os.makedirs(save_path, exist_ok=True)
    
    print("🎯 Engaging Apex Training Loop (2M Steps)...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # 5. Persistent Storage
    model_file = f"{save_path}/{symbol}_{timeframe}_apex_v6_expert"
    stats_file = f"{save_path}/{symbol}_{timeframe}_apex_v6_vec_normalize.pkl"
    
    model.save(model_file)
    env.save(stats_file)
    
    print(f"🏁 Apex Training Complete. Model fused to {model_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd")
    parser.add_argument("--steps", type=int, default=2000000)
    args = parser.parse_args()
    
    train_apex_sniper(symbol=args.symbol, total_timesteps=args.steps)
