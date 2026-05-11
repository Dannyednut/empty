"""
 Sniper V4 Evaluation Script
Detailed performance analysis for the 2D action model.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data.fetch_mt5 import fetch_data
from env.trading_env_v4 import SniperTradingEnv
from features.indicators_v2 import build_features, get_feature_columns

def evaluate_v4(symbol="xauusd", timeframe="m5"):
    model_path = f"models/{symbol}/sniper/{symbol}_{timeframe}_sniper_v4_expert.zip"
    stats_path = f"models/{symbol}/sniper/{symbol}_{timeframe}_sniper_v4_vec_normalize.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Load Data (Evaluation Set)
    if not os.path.exists(f"data/{symbol}_{timeframe}.csv"):
        fetch_data(symbol, timeframe)
    df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
    base_tf = '5min' if timeframe == 'm5' else '15min' if timeframe == 'm15' else '30min'
    df = build_features(df, base_timeframe=base_tf)
    feature_cols = get_feature_columns()
    
    # Use last 10,000 bars for evaluation
    eval_df = df.tail(5000).copy()
    
    # 2. Setup Environment
    def make_env():
        return SniperTradingEnv(df=eval_df, feature_columns=feature_cols)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    # 3. Predict
    model = PPO.load(model_path)
    obs = env.reset()
    
    values = []
    positions = []
    multipliers = []
    
    print("Running Sniper V4 Backtest...")
    for _ in range(len(eval_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        act_flat = action.flatten()
        positions.append(act_flat[0])
        multipliers.append(act_flat[1])
        values.append(info[0]['portfolio_value'])
        
        if done: break

    # 4. Metrics
    final_return = (values[-1] - 10000) / 10000
    print(f"\nEVALUATION RESULTS:")
    print(f"   Final Return: {final_return:.2%}")
    print(f"   Avg ATR Multiplier: {np.mean(multipliers):.2f}x")
    print(f"   Max Position Conviction: {np.max(np.abs(positions)):.2%}")
    
    # 5. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    ax1.plot(values, label='Equity Curve', color='cyan')
    ax1.set_title(f"Sniper V4 Performance: {symbol.upper()}")
    ax1.legend()
    
    ax2.fill_between(range(len(positions)), positions, color='lime', alpha=0.3, label='Directional Bias')
    ax2.set_title("Position Conviction (-1 to 1)")
    ax2.legend()
    
    ax3.plot(multipliers, color='orange', label='Dynamic Multiplier')
    ax3.axhline(y=1.5, color='white', linestyle='--', alpha=0.5)
    ax3.set_title("Learned ATR Multiplier (Dynamic Stops)")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    report_path = f'reports/sniper_v4_{symbol}_{timeframe}_eval.png'
    plt.savefig(report_path)
    print(f"Evaluation chart saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd")
    parser.add_argument("--timeframe", type=str, default="m5")
    args = parser.parse_args()
    
    evaluate_v4(symbol=args.symbol, timeframe=args.timeframe)
