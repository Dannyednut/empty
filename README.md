# 🎯 PPO Sniper - Institutional Trading Suite

Advanced reinforcement learning trading suite using Proximal Policy Optimization (PPO). This platform has evolved from a research project into a fully automated, multi-asset institutional trading system.

## 🚀 The Sniper Ecosystem

This suite consists of four specialized layers designed for high-performance trading of Gold (XAUUSD), Silver (XAGUSD), and Oil (WTI).

### 1. Performance Dashboard (`dashboard.py`)
Generates a stunning, interactive HTML report of your live trading history.
- **Run**: `py -3.11 dashboard.py`
- **Output**: Open `results/live_dashboard.html` in any browser.
- **Metrics**: Real-time Win Rate, Profit Factor, and Max Drawdown tracking.

### 2. Auto-Retraining Pipeline (`retrain_pipeline.py`)
Features the **"Champion vs. Challenger"** validation system to prevent model decay.
- **Run**: `py -3.11 retrain_pipeline.py`
- **Logic**: Automatically fetches last 30 days of data -> Hides latest 5 days (Hold-out set) -> Fine-tunes PPO -> Only promotes the new model if it outperforms the current live version on unseen data.

### 3. Universal Expert Trainer (`universal_trainer.py`)
Master any market by training a specialized brain for that specific asset.
- **Run**: `py -3.11 universal_trainer.py [SYMBOL] [STEPS]`
- **Example**: `py -3.11 universal_trainer.py XAGUSD 500000`
- **Result**: Creates an expert model in `models/[symbol]/experts/`.

### 4. Multi-Instance Manager (`multi_instance_trader.py`)
The ultimate execution bridge that manages multiple experts simultaneously.
- **Run**: `py -3.11 multi_instance_trader.py`
- **Logic**: Independent conviction-drop exit tracking, dynamic lot sizing, and multi-symbol order management through a single MT5 connection.

---

## 🏗️ Project Structure

```
PPO_trader/
├── models/                    # 🧠 Brains & Stats
│   └── [symbol]/              # e.g., xauusd/
│       ├── experts/           # Production models
│       ├── checkpoints/       # Training snapshots
│       └── backups/           # Replaced versions
│
├── data/                      # 📊 Raw Market Data
├── features/                  # 🛠️ 35+ Indicator Engineering
├── env/                       # 🌍 Advanced Gymnasium Environments
├── evaluation/                # ⚖️ Backtesting & Metrics
├── logs/                      # 📈 TensorBoard & Live CSV Logs
├── results/                   # 🖼️ Dashboards & PNG Reports
│
├── live_trader_mt5_v2.py      # Single-asset Gold Live Trader
├── multi_instance_trader.py   # Multi-asset Execution Manager
├── retrain_pipeline.py        # Automated Maintenance
├── universal_trainer.py       # Specialist Training 
└── dashboard.py               # Performance Visualization 
```

---

## 🔧 Technical Core

### Feature Engineering (`indicators_v2.py`)
Over **35 technical features** providing a deep market context:
- **Momentum**: RSI, MACD, ADX, Stochastic.
- **Volatility**: ATR, Historical Volatility, BB Width.
- **Trend**: SMMA, TEMA, Multi-timeframe RSI (15M/30M).
- **Context**: Market Sessions (Asian/London/NY), Hour/Day Sin/Cos encoding.

### High-Frequency Env (`trading_env_v3.py`)
- **Continuous Action Space**: Precise position sizing from -100% to +100%.
- **Net Accounting**: Robust cash/share tracking to prevent numerical instability.
- **Safety**: 0.5% Hard Stop-Loss and 1.5% Take-Profit by default.

---

## 📈 Strategic Workflow

1. **Research**: Train a new expert using `universal_trainer.py`.
2. **Deploy**: Add the model path to `ASSETS` in `multi_instance_trader.py`.
3. **Analyze**: Run `dashboard.py` weekly to review the equity curve and drawdown.
4. **Maintain**: Run `retrain_pipeline.py` monthly to adapt the model to current market conditions.

---

## 🎓 Training Signs

**Good Signs:**
✅ `ep_rew_mean` increasing over time.
✅ `explained_variance` > 0.5 (Value function understands the market).
✅ `eval/sharpe_ratio` > 1.5 during retraining duels.

**Bad Signs:**
❌ `approx_kl` > 0.05 (Updates are too aggressive, reducing LR).
❌ `entropy_loss` flatlining too early (The model stopped exploring patterns).

---

## ⚠️ Disclaimer

This is a professional algorithmic trading suite. 
**Past performance does not guarantee future results.** 
Always verify in **Demo Mode** (`DRY_RUN = True`) before risking real capital. The authors are not responsible for financial losses incurred.

---
**Built with ❤️ for High-Alpha Algorithmic Trading**
