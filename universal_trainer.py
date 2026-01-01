"""
PPO Trading Sniper - Universal Master Trainer
Allows training specialized expert models for any financial symbol.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Import project modules
sys.path.append('.')
from env.trading_env_v3 import AdvancedTradingEnv
from features.indicators_v2 import build_features, get_feature_columns

def train_expert(symbol, timesteps=500000, timeframe=mt5.TIMEFRAME_M5):
    print(f"🚀 Initializing Master Training for {symbol}...")
    
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return
        
    # 1. Data Fetching (Use last 100 days for broad master training)
    print(f"📡 Fetching historical data for {symbol}...")
    utc_from = datetime.now() - timedelta(days=100)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, datetime.now())
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {symbol}. Ensure the symbol is visible in MT5 MarketWatch.")
        return
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 2. Build Pipeline
    print("🛠️ Building Features...")
    df = build_features(df, base_timeframe='5min').dropna()
    
    # 3. Setup Environment
    print("🌍 Preparing Environment...")
    def make_env():
        return AdvancedTradingEnv(df, initial_balance=10000)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 4. Model Definition
    print("🧠 Defining Neural Architecture...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=8192,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./logs/tensorboard/{symbol.lower()}_expert/"
    )
    
    # 5. Execute Training
    print(f"⚡ Executing {timesteps} steps of training...")
    
    # Create structure
    symbol_root = os.path.join("models", symbol.lower())
    expert_dir = os.path.join(symbol_root, "experts")
    checkpoint_dir = os.path.join(symbol_root, "checkpoints")
    
    os.makedirs(expert_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="expert"
    )
    
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    # 6. Final Save
    final_name = f"{symbol.lower()}_m5_ppo_expert"
    model.save(os.path.join(expert_dir, final_name))
    env.save(os.path.join(expert_dir, f"{final_name}_vec_normalize.pkl"))
    
    print(f"✅ Training Complete! Model saved as {expert_dir}/{final_name}.zip")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python universal_trainer.py SYMBOL [steps]")
        sys.exit(1)
        
    target_symbol = sys.argv[1].upper()
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
    
    train_expert(target_symbol, timesteps=steps)
