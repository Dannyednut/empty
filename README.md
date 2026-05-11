# PPO Sniper - Institutional Trading Ecosystem

Advanced reinforcement learning trading suite using Proximal Policy Optimization (PPO). This platform features a dual-engine architecture (**Sniper V4** for Scalping and **Expert V3** for Sessioned Trading), managed through a unified, high-performance web dashboard.

## Key Evolutionary Features (v2.2)
- **Dual-Model Architecture**: Switch between high-frequency Sniper (Adaptive ATR) and traditional Expert models.
- **PPO Command Center**: Glassmorphism web dashboard for Training, Evaluation, and Live Fleet management.
- **Adaptive Sniper Engine**: Sniper V4 uses a 2D action space (Direction + Volatility Multiplier) for dynamic SL/TP calculation.
- **Mission Telemetry**: Real-time log streaming from active background tasks (Python 3.9 unbuffered).
- **War Room**: Full evaluation suite with automated PNG report generation and gallery viewing.
- **Portfolio Fleet**: Deploy multiple specialized models across different assets (Gold, Silver, Oil) from a single bridge.

## Getting Started

### 1. Installation
Ensure MT5 is installed and logged into your account.
```bash
pip install -r requirements.txt
pip install "numpy<2.0"  # Critical for MetaTrader5 compatibility
```

### 2. Launch the Ecosystem
Start the master Command Center:
```powershell
py -3.9 api_server_v2.py
```
Open **`http://localhost:8000`** in your browser.

---

## The Ecosystem Layers

### The Forge (Training)
Train specialized brains for any symbol/timeframe.
- **Expert V3**: Traditional PPO with 3-action space (Buy, Sell, Hold).
- **Sniper V4**: Advanced 2D output (Position Conviction + ATR Multiplier).
- **Execution**: Managed via `train_v3.py` and `train_v4.py`.

### The Evaluation
Rigorous backtesting with visual reporting.
- generates **Equity Curves**, **Position Heatmaps**, and **ATR Volatility** diagnostics.
- Reports are saved to `reports/` and viewable in the dashboard gallery.

### Live Deployment
Deploy models to the markets with surgical precision.
- **Live Mode**: Executes trades directly on your MT5 account.
- **Dry Run (Paper)**: Institutional-grade monitoring mode to verify model conviction before risking capital.
- **Fleet Manager**: Orchestrates `multi_instance_trader.py` for portfolio-wide execution.

---

## Project Architecture
```
PPO_trader/
├── api_server_v2.py           # Master Hub (FastAPI + Real-time Logs)
├── templates/index.html       # Glassmorphism Dashboard UI
├── models/                    # Brains & Normalization Stats
│   └── [symbol]/              # Hierarchical model storage
├── data/                      # Raw csv Market Data
├── features/                  # 35+ Indicator Engineering (Indicators V2)
├── env/                       # Advanced Gymnasium Environments (V3 & V4)
├── reports/                   # Evaluation Charts & Results
├── live_sniper_v4.py          # Scalper Execution Bridge
├── live_trader_mt5_v2.py      # Sessioned Execution Bridge
└── multi_instance_trader.py   # Multi-asset Portfolio Manager
```

---

## Expert vs. Sniper: What's the Difference?

| Feature | Expert V3 (Sessioned) | Sniper V4 (Adaptive) |
| :--- | :--- | :--- |
| **Logic** | Fixed SL/TP Percentages | Adaptive ATR-based SL/TP |
| **Action Space** | 1D (Buy/Sell/Wait) | 2D (Position + Multiplier) |
| **Exit Trigger** | Conviction-Drop Logic | Continuous Neural Adjustment |
| **Ideal For** | Major Session Trends | High-Volatility Scalping |

---

## Safe Deployment Protocol
1. **Train** in The Forge for 1,000,000 steps.
2. **Evaluate** in The War Room (Check for > 1.5 Sharpe Ratio).
3. **Dry Run** for at least 24 hours to verify "Live Telemetry" parity.
4. **Go Live** with minimal lot sizes (0.01) to verify fill performance.

---

## Disclaimer
 algorithmic trading carries high risk. **Past performance does not guarantee future results.** The authors are not responsible for financial losses. Always use `DRY_RUN = True` for initial testing.

---
**institutional-Grade Alpha via Reinforcement Learning.**
