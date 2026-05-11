"""
PPO Trading Sniper - Auto-Retraining Pipeline V3
Institutional "Champion vs. Challenger" Model Validation
Supports both Experts (V3) and Snipers (V4)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import sys
import shutil
import argparse
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import project modules
sys.path.append('.')
from env.trading_env_v3 import AdvancedTradingEnv
from env.trading_env_v4 import SniperTradingEnv
from features.indicators_v2 import build_features, get_feature_columns

def fetch_latest_data(symbol, timeframe_mt5, days=30):
    print(f"📡 Fetching last {days} days of data for {symbol}...")
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return None
        
    utc_from = datetime.now() - timedelta(days=days)
    rates = mt5.copy_rates_from(symbol, timeframe_mt5, utc_from, datetime.now())
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"❌ No data received for {symbol}")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"✅ Fetched {len(df)} candles.")
    
    # Build features
    tf_str = '5min' if timeframe_mt5 == mt5.TIMEFRAME_M5 else '15min' if timeframe_mt5 == mt5.TIMEFRAME_M15 else '30min'
    df = build_features(df, base_timeframe=tf_str).dropna()
    return df

def get_env_creator(model_type, data, feature_cols):
    if model_type == "sniper":
        return lambda: SniperTradingEnv(data, feature_columns=feature_cols, initial_cash=10000)
    else:
        return lambda: AdvancedTradingEnv(data, feature_columns=feature_cols, initial_cash=10000)

def backtest_model(model_path, stats_path, model_type, data, feature_cols):
    """Run a simulated backtest to get Total Profit"""
    make_env = get_env_creator(model_type, data, feature_cols)
    env = DummyVecEnv([make_env])
    
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
    # Get final equity from the internal env
    final_equity = info[0]['portfolio_value']
    profit = final_equity - 10000
    return profit

def run_retrain_duel(args, data):
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    # 1. Split Data
    data = data.sort_values('time')
    split_idx = data[data['time'] < (data['time'].max() - timedelta(days=args.holdout))].index.max()
    
    train_data = data.loc[:split_idx].copy()
    holdout_data = data.loc[split_idx+1:].copy()
    
    print(f"✂️ Data Split: Train={len(train_data)} bars | Holdout={len(holdout_data)} bars")

    # 2. Setup Paths
    symbol_low = args.symbol.lower()
    sub_folder = "sniper" if args.model_type == "sniper" else "experts"
    model_dir = f"models/{symbol_low}/{sub_folder}"
    backup_dir = f"models/{symbol_low}/backups"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    
    if args.model_type == "sniper":
        model_name = f"{symbol_low}_{args.timeframe}_sniper_v4_expert"
    else:
        model_name = f"{symbol_low}_{args.timeframe}_ppo_expert"
        
    model_path = os.path.join(model_dir, f"{model_name}.zip")
    stats_path = os.path.join(model_dir, f"{model_name}_vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Base model not found at {model_path}.")
        return False

    # 3. Train the Challenger
    print(f"🧠 Fine-tuning Challenger ({args.model_type.upper()})...")
    make_train_env = get_env_creator(args.model_type, train_data, feature_cols)
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize.load(stats_path, train_env)
    train_env.training = True
    
    challenger_model = PPO.load(model_path, env=train_env)
    challenger_model.learn(total_timesteps=args.steps)
    
    # Save Challenger Temporary
    chall_path = os.path.join(model_dir, "challenger_temp.zip")
    chall_stats = os.path.join(model_dir, "challenger_temp_stats.pkl")
    challenger_model.save(chall_path)
    train_env.save(chall_stats)

    # 4. THE DUEL
    print("\n⚔️ STARTING DUEL ON HOLDOUT DATA...")
    
    print("🛡️ Champion (Current) testing...")
    champ_profit = backtest_model(model_path, stats_path, args.model_type, holdout_data, feature_cols)
    
    print("🗡️ Challenger (New) testing...")
    chall_profit = backtest_model(chall_path, chall_stats, args.model_type, holdout_data, feature_cols)
    
    print(f"\n📊 DUEL RESULTS:")
    print(f"🏆 Champion Profit: ${champ_profit:.2f}")
    print(f"🚀 Challenger Profit: ${chall_profit:.2f}")

    if chall_profit > champ_profit:
        print("✅ CHALLENGER WINS! Proceeding to promotion.")
        # Promote
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Backup old
        shutil.copy2(model_path, os.path.join(backup_dir, f"{model_name}_backup_{timestamp}.zip"))
        # Move new in
        shutil.move(chall_path, model_path)
        shutil.move(chall_stats, stats_path)
        print(f"✅ Production model updated: {model_path}")
        return True
    else:
        print("❌ CHALLENGER FAILED to beat Champion. Retaining current model.")
        if os.path.exists(chall_path): os.remove(chall_path)
        if os.path.exists(chall_stats): os.remove(chall_stats)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Sniper/Expert Auto-Retrain Pipeline")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--timeframe", type=str, default="m5")
    parser.add_argument("--model_type", type=str, default="expert", choices=["expert", "sniper"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=30)
    args = parser.parse_args()
    
    tf_mt5 = getattr(mt5, f"TIMEFRAME_{args.timeframe.upper()}")
    
    print(f"🔄 MAINTENANCE MODE: {args.model_type.upper()} | {args.symbol} | {args.timeframe}")
    data = fetch_latest_data(args.symbol, tf_mt5, args.lookback)
    
    if data is not None:
        run_retrain_duel(args, data)
    
    print("🏁 Pipeline Finished.")
