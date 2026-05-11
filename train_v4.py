"""
PPO Training Script v4 (The Sniper)
Trains an agent on the SniperTradingEnv (v4)
Features: Multi-dimensional actions and adaptive volatility stops.
"""
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from data.fetch_mt5 import fetch_data
from env.trading_env_v4 import SniperTradingEnv
from features.indicators_v2 import build_features, get_feature_columns

def train_sniper(symbol="xauusd", timeframe="m5", total_timesteps=500_000):
    print(f"Starting Sniper (V4) Training for {symbol.upper()}...")
    
    # 1. Load and Prepare Data
    data_path = f"data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        fetch_data(symbol, timeframe)
    
    df = pd.read_csv(data_path)
    # Ensure manual normalization is DISABLED in indicators_v2 (handled by VecNormalize)
    base_tf = '5min' if timeframe == 'm5' else '15min' if timeframe == 'm15' else '30min'
    df = build_features(df, base_timeframe=base_tf)
    feature_cols = get_feature_columns()
    
    # 2. Create Environment
    def make_env():
        env = SniperTradingEnv(
            df=df,
            feature_columns=feature_cols,
            initial_cash=10_000,
            episode_length=2000
        )
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    
    # Normalization (Crucial for V4)
    # We now allow it to normalize the raw indicators
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. PPO Model Configuration
    # Using a slightly wider network for the 2D action space
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 64], qf=[128, 128, 64])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-5, # Slightly slower for better convergence on 2D actions
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Increased entropy to explore the ATR Multiplier space
        tensorboard_log=f"./logs/sniper_v4_{symbol}/",
        device="auto"
    )

    # 4. Callbacks
    save_path = f"models/{symbol}/sniper/"
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_path,
        name_prefix=f"{symbol}_sniper_v4"
    )

    # 5. Execute Training
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 6. Save Final Model and Stats
    model.save(f"{save_path}/{symbol}_sniper_v4_expert")
    env.save(f"{save_path}/{symbol}_sniper_v4_vec_normalize.pkl")
    
    print(f"Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd") 
    parser.add_argument("--timeframe", type=str, default="m5")
    parser.add_argument("--steps", type=int, default=500_000)
    args = parser.parse_args()
    
    train_sniper(symbol=args.symbol, timeframe=args.timeframe, total_timesteps=args.steps)
