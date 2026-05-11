"""
PPO Sniper V5 - Pulse Engine Evaluator
Simulates intra-bar decision making and generates performance diagnostics.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.trading_env_v5_pulse import SniperPulseEnvV5
from features.indicators_v2 import build_features, get_feature_columns

def evaluate_pulse_v5(symbol="xauusd", timeframe="m1", plot=True):
    print(f"📊 Starting Sniper V5 Pulse Evaluation for {symbol.upper()}...")
    
    # 1. Load Data
    data_path = f"data/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path):
        print("❌ Error: Evaluation data not found.")
        return
        
    df = pd.read_csv(data_path).iloc[-5000:] # Last 5000 M1 bars
    df = build_features(df, base_timeframe='1min').dropna()
    feature_columns = get_feature_columns()
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # 2. Setup Environment
    model_path = f"models/{symbol.lower()}/sniper_v5/{symbol}_{timeframe}_pulse_v5_expert.zip"
    stats_path = f"models/{symbol.lower()}/sniper_v5/{symbol}_{timeframe}_pulse_v5_vec_normalize.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    def make_eval_env():
        return SniperPulseEnvV5(df, feature_columns, initial_cash=10000)
    
    env = DummyVecEnv([make_eval_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    # 3. Predict Pulses
    model = PPO.load(model_path)
    obs = env.reset()
    
    history = {
        'equity': [],
        'position': [],
        'multiplier': [],
        'pulse': []
    }
    
    print("🛰️ Running Pulse Backtest...")
    # Each bar has 3 pulses, so we run for len(df)*3 steps
    total_steps = (len(df) - 100) * 3 
    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        act_flat = action.flatten()
        history['equity'].append(info[0]['portfolio_value'])
        history['position'].append(act_flat[0])
        history['multiplier'].append(act_flat[1])
        history['pulse'].append(info[0]['pulse'])
        
        if done: break

    # 4. Generate Report
    print(f"\n📈 PULSE PERFORMANCE REPORT ({symbol.upper()}):")
    final_return = (history['equity'][-1] - 10000) / 10000
    print(f"   Final Return: {final_return:.2%}")
    print(f"   Avg Conviction: {np.mean(np.abs(history['position'])):.2%}")
    print(f"   Avg Pulse Multiplier: {np.mean(history['multiplier']):.2f}x")

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Equity Curve
        ax1.plot(history['equity'], color='#00ff88', label='Pulse Equity', linewidth=1.5)
        ax1.set_title(f"Sniper V5 Pulse Performance - {symbol.upper()} M1", color='white', fontsize=14)
        ax1.set_ylabel("Portfolio Value ($)", color='white')
        ax1.grid(True, alpha=0.1)
        ax1.legend()
        
        # Conviction & Pulses
        ax2.fill_between(range(len(history['position'])), history['position'], color='cyan', alpha=0.3, label='Neural Conviction')
        ax2.set_ylabel("Position / Context", color='white')
        ax3 = ax2.twinx()
        ax3.plot(history['multiplier'], color='orange', alpha=0.5, label='ATR Mult')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper right')
        
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#333')
            
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/sniper_v5_{symbol}_{timeframe}_pulse_report.png"
        plt.savefig(report_path)
        print(f"✅ Report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="xauusd")
    parser.add_argument("--timeframe", type=str, default="m1")
    args = parser.parse_args()
    
    evaluate_pulse_v5(symbol=args.symbol)
