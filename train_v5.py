"""
PPO Sniper V5 - High-Frequency "Pulse" Trainer
Focused on Intra-Bar Alpha and M1 Patterns.
"""
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.trading_env_v5_pulse import SniperPulseEnvV5
from features.indicators_v2 import build_features, get_feature_columns
from data.fetch_mt5 import fetch_data

def train_pulse_sniper(symbol="xauusd", timeframe="m1", total_timesteps=1_000_000):
    print(f"⚡ Initializing Sniper Pulse (V5) Training for {symbol.upper()}...")
    
    # 1. Prepare High-Res Data (M1)
    data_path = f"data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        print(f"📡 Fetching fresh {timeframe.upper()} data...")
        df = fetch_data(symbol=symbol.upper(), timeframe=timeframe, bars=99999)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    print(f"📊 Building High-Freq Features...")
    df = build_features(df, base_timeframe='1min')
    df = df.dropna()
    
    feature_columns = get_feature_columns()
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # 2. Setup High-Freq Environment
    def make_env():
        return SniperPulseEnvV5(
            df=df,
            feature_columns=feature_columns,
            episode_length=1440 # 1 day of M1
        )
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 3. High-Frequency PPO Hyperparameters
    # We use a smaller learning rate and larger batch for high-noise M1 data
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-5, # Conservative for high-noise
        n_steps=4096,      # Multiple episodes worth of pulses
        batch_size=512,
        n_epochs=10,
        gamma=0.98,        # Focus on near-term scalps
        gae_lambda=0.92,
        clip_range=0.15,
        ent_coef=0.015,     # Encourage more exploration in noise
        tensorboard_log="./logs/sniper_v5_pulse/"
    )
    
    # 4. Mission Execution
    save_path = f"models/{symbol.lower()}/sniper_v5"
    os.makedirs(save_path, exist_ok=True)
    
    print("🎯 Engaging Pulse Training Loop...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # 5. Persistent Storage
    model_file = f"{save_path}/{symbol}_{timeframe}_pulse_v5_expert"
    stats_file = f"{save_path}/{symbol}_{timeframe}_pulse_v5_vec_normalize.pkl"
    
    model.save(model_file)
    env.save(stats_file)
    
    print(f"🏁 Pulse Training Complete. Model fused to {model_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd")
    parser.add_argument("--timeframe", type=str, default="m1")
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()
    
    train_pulse_sniper(symbol=args.symbol, total_timesteps=args.steps)
