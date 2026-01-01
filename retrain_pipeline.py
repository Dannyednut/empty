"""
PPO Trading Sniper - Auto-Retraining Pipeline V2
Institutional "Champion vs. Challenger" Model Validation
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import sys
import shutil
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import project modules
sys.path.append('.')
from env.trading_env_v3 import AdvancedTradingEnv
from features.indicators_v2 import build_features, get_feature_columns

# =============================================================
# CONFIGURATION
# =============================================================
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK_DAYS = 30
HOLDOUT_DAYS = 5      # Days to use for the "Duel" validation
TRAINING_STEPS = 50000 

# New Organized Paths
SYMBOL_ROOT = os.path.join("models", SYMBOL.lower())
MODEL_DIR = os.path.join(SYMBOL_ROOT, "experts")
BACKUP_DIR = os.path.join(SYMBOL_ROOT, "backups")
MODEL_NAME = f"{SYMBOL.lower()}_m5_ppo_expert"

def fetch_latest_data(days=30):
    print(f"📡 Fetching last {days} days of data for {SYMBOL}...")
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return None
        
    utc_from = datetime.now() - timedelta(days=days)
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, utc_from, datetime.now())
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("❌ No data received")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"✅ Fetched {len(df)} candles.")
    
    # Build features
    df = build_features(df, base_timeframe='5min').dropna()
    return df

def backtest_model(model_path, stats_path, data):
    """Run a simulated backtest to get Total Profit"""
    def make_env():
        return AdvancedTradingEnv(data, initial_balance=10000)
    
    env = DummyVecEnv([make_env])
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
    # Get final equity from the internal env
    final_equity = env.get_attr('equity')[0]
    profit = final_equity - 10000
    return profit, total_reward

def run_retrain_duel(data):
    # 1. Split Data
    data = data.sort_values('time')
    split_idx = data[data['time'] < (data['time'].max() - timedelta(days=HOLDOUT_DAYS))].index.max()
    
    train_data = data.loc[:split_idx].copy()
    holdout_data = data.loc[split_idx+1:].copy()
    
    print(f"✂️ Data Split: Train={len(train_data)} bars | Holdout={len(holdout_data)} bars")

    # 2. Train the Challenger
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")
    stats_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        print("❌ Base model not found.")
        return False

    print("🧠 Fine-tuning Challenger...")
    def make_train_env(): return AdvancedTradingEnv(train_data, initial_balance=10000)
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize.load(stats_path, train_env)
    train_env.training = True
    
    challenger_model = PPO.load(model_path, env=train_env)
    challenger_model.learn(total_timesteps=TRAINING_STEPS)
    
    # Save Challenger Temporary
    chall_path = os.path.join(MODEL_DIR, "challenger_temp.zip")
    chall_stats = os.path.join(MODEL_DIR, "challenger_temp_stats.pkl")
    challenger_model.save(chall_path)
    train_env.save(chall_stats)

    # 3. THE DUEL
    print("\n⚔️ STARTING DUEL ON HOLDOUT DATA...")
    
    print("🛡️ Champion (Current) testing...")
    champ_profit, _ = backtest_model(model_path, stats_path, holdout_data)
    
    print("🗡️ Challenger (New) testing...")
    chall_profit, _ = backtest_model(chall_path, chall_stats, holdout_data)
    
    print(f"\n📊 RESULTS:")
    print(f"🏆 Champion Profit: ${champ_profit:.2f}")
    print(f"🚀 Challenger Profit: ${chall_profit:.2f}")

    if chall_profit > champ_profit:
        print("✅ CHALLENGER WINS! Proceeding to promotion.")
        # Move temp to challenger perm
        shutil.move(chall_path, os.path.join(MODEL_DIR, f"{MODEL_NAME}_challenger.zip"))
        shutil.move(chall_stats, os.path.join(MODEL_DIR, f"{MODEL_NAME}_challenger_vec_normalize.pkl"))
        return True
    else:
        print("❌ CHALLENGER FAILED to beat Champion. Retaining current model.")
        os.remove(chall_path)
        os.remove(chall_stats)
        return False

def promote_model():
    print("🏆 Promoting Challenger to Production...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    shutil.copy2(os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip"), 
                 os.path.join(BACKUP_DIR, f"{MODEL_NAME}_{timestamp}.zip"))
    
    shutil.move(os.path.join(MODEL_DIR, f"{MODEL_NAME}_challenger.zip"), 
                os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip"))
    shutil.move(os.path.join(MODEL_DIR, f"{MODEL_NAME}_challenger_vec_normalize.pkl"), 
                os.path.join(MODEL_DIR, f"{MODEL_NAME}_vec_normalize.pkl"))
    print("✅ Promotion Complete.")

if __name__ == "__main__":
    print("🔄 PPO SNIPER AUTO-RETRAIN PIPELINE V2")
    data = fetch_latest_data(LOOKBACK_DAYS)
    if data is not None:
        if run_retrain_duel(data):
            promote_model()
    print("🏁 Pipeline Finished.")
