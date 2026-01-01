"""
PPO Trading Sniper - Performance Dashboard
Generates an interactive HTML report from live trading logs
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# =============================================================
# CONFIGURATION
# =============================================================
LOG_FILES = ["logs/live_trading_log.csv", "logs/live_trading_v2_log.csv", "logs/multi_asset_live_log.csv"]
OUTPUT_FILE = "results/live_dashboard.html"

def load_and_clean_logs(file_paths):
    dfs = []
    for path in file_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
                # Defaults for columns that might be missing in older logs
                if 'peak_conv' not in df.columns: df['peak_conv'] = 0.0
                if 'symbol' not in df.columns: df['symbol'] = 'XAUUSD'
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    if not dfs:
        return None
        
    df = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_metrics(df):
    # Filter for trade closure events to calculate real PnL
    trades = df[df['trade_type'] == 'CLOSE'].copy()
    
    if trades.empty:
        return {
            "Total Trades": 0,
            "Win Rate": "0%",
            "Profit Factor": "0.00",
            "Final Equity": f"${df['equity'].iloc[-1]:.2f}" if not df.empty else "$0.00",
            "Max Drawdown": "0.0%"
        }
    
    # Simple metric: how many times does equity increase vs decrease after a close?
    equity_diffs = df['equity'].diff().dropna()
    wins = equity_diffs[equity_diffs > 0]
    losses = equity_diffs[equity_diffs < 0]
    
    win_rate = (len(wins) / (len(wins) + len(losses))) * 100 if (len(wins) + len(losses)) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    # Drawdown
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak
    max_dd = drawdown.min() * 100
    
    return {
        "Total Trades": len(trades),
        "Win Rate": f"{win_rate:.1f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Final Equity": f"${df['equity'].iloc[-1]:.2f}",
        "Max Drawdown": f"{max_dd:.1f}%"
    }

def create_dashboard(df, metrics):
    if df is None:
        print("No data to visualize.")
        return

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity Curve", "Model Confidence (Conviction)", "Drawdown (%)")
    )

    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['equity'], name="Equity", 
                   line=dict(color='#00ff88', width=3), fill='tozeroy'),
        row=1, col=1
    )

    # 2. Confidence / Action
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['action'], name="Conviction",
                   line=dict(color='#00ccff', width=1.5), mode='lines'),
        row=2, col=1
    )
    # Add logic lines (Neutral Zone)
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange", opacity=0.3, row=2, col=1)
    fig.add_hline(y=-0.2, line_dash="dash", line_color="orange", opacity=0.3, row=2, col=1)

    # 3. Drawdown
    peak = df['equity'].cummax()
    dd = (df['equity'] - peak) / peak * 100
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=dd, name="Drawdown", 
                   fill='tozeroy', line=dict(color='#ff4444')),
        row=3, col=1
    )

    # Add annotations for Metrics
    metric_text = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    
    fig.update_layout(
        title=dict(
            text=f"PPO Sniper Performance Dashboard<br><sup>{metric_text}</sup>",
            x=0.5, font=dict(size=24, color='white')
        ),
        template="plotly_dark",
        height=900,
        showlegend=False,
        paper_bgcolor='black',
        plot_bgcolor='#111111'
    )

    # Ensure result directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fig.write_html(OUTPUT_FILE)
    print(f"Dashboard generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    print("Generating Live Dashboard...")
    df = load_and_clean_logs(LOG_FILES)
    if df is not None:
        metrics = calculate_metrics(df)
        create_dashboard(df, metrics)
    else:
        print("No log files found. Start the trader first!")
