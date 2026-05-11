"""
PPO Sniper V6 - Apex Survivalist Evaluator
Simulates reversal avoidance and early exit alpha.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.trading_env_v6_apex import SniperApexEnvV6
from features.indicators_v2 import build_features, get_feature_columns

def evaluate_apex_v6(symbol="xauusd", timeframe="m1", plot=True):
    print(f"📊 Starting Sniper V6 Apex Evaluation for {symbol.upper()}...")
    
    # 1. Load Data
    data_path = f"data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        print("❌ Error: Evaluation data not found.")
        return
        
    df = pd.read_csv(data_path).iloc[-5000:] 
    df = build_features(df, base_timeframe='1min').dropna()
    feature_columns = get_feature_columns()
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # 2. Setup Environment
    model_path = f"models/{symbol.lower()}/sniper_v6/{symbol}_{timeframe}_apex_v6_expert.zip"
    stats_path = f"models/{symbol.lower()}/sniper_v6/{symbol}_{timeframe}_apex_v6_vec_normalize.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    def make_eval_env():
        return SniperApexEnvV6(df, feature_columns, initial_cash=10000)
    
    env = DummyVecEnv([make_eval_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    # 3. Predict Apex Actions
    model = PPO.load(model_path)
    obs = env.reset()
    
    history = {
        'equity': [],
        'position': [],
        'flush': [],
        'stagnation': []
    }
    
    print("🛰️ Running Apex Survival Backtest...")
    total_steps = (len(df) - 100) * 3 
    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        act_flat = action.flatten()
        history['equity'].append(info[0]['portfolio_value'])
        history['position'].append(act_flat[0])
        history['flush'].append(act_flat[2])
        history['stagnation'].append(info[0]['stagnation'])
        
        if done: break

    # 4. Generate Apex Report
    print(f"\n📈 APEX SURVIVAL REPORT ({symbol.upper()}):")
    final_return = (history['equity'][-1] - 10000) / 10000
    print(f"   Final Return: {final_return:.2%}")
    print(f"   Max Stagnation Tolerance: {np.max(history['stagnation']):.1f} pulses")
    print(f"   Avg Neural Panic (Flush): {np.mean(history['flush']):.2%}")

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Equity Curve
        ax1.plot(history['equity'], color='#f0ad4e', label='Apex Equity', linewidth=2.0)
        ax1.set_title(f"Sniper V6 Apex Survivalist - {symbol.upper()} M1", color='white', fontsize=14)
        ax1.set_ylabel("Portfolio Value ($)", color='white')
        ax1.grid(True, alpha=0.1)
        ax1.legend()
        
        # Conviction & Flush Actions
        ax2.fill_between(range(len(history['position'])), history['position'], color='#00d1b2', alpha=0.4, label='Neural Conviction')
        ax2.plot(history['flush'], color='red', alpha=0.6, label='Panic Exit (Flush)')
        ax2.set_ylabel("Neural Dynamics", color='white')
        
        ax3 = ax2.twinx()
        ax3.plot(history['stagnation'], color='white', alpha=0.2, label='Stagnated Time')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper right')
        
        fig.patch.set_facecolor('#0d1117')
        ax1.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#30363d')
            
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/sniper_v6_{symbol}_{timeframe}_apex_report.png"
        plt.savefig(report_path)
        print(f"✅ Apex Report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd")
    parser.add_argument("--timeframe", type=str, default="m1")
    args = parser.parse_args()
    
    evaluate_apex_v6(symbol=args.symbol)
