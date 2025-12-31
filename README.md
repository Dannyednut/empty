# PPO Trading Agent - Enhanced Version

Advanced reinforcement learning trading agent using Proximal Policy Optimization (PPO) for XAU/USD (Gold) trading.

## 🎯 Features

### Enhanced v3 Implementation
- ✅ **35+ Technical Indicators** - Comprehensive feature engineering
- ✅ **Multi-Timeframe Analysis** - 5M, 15M, 30M context
- ✅ **Continuous Action Space** - Position sizing (0-100%)
- ✅ **Risk Management** - Stop-loss, take-profit, drawdown limits
- ✅ **Risk-Adjusted Rewards** - Sharpe ratio optimization
- ✅ **GPU Acceleration** - CUDA support for faster training
- ✅ **TensorBoard Logging** - Real-time training monitoring
- ✅ **Model Checkpointing** - Save best models automatically
- ✅ **Comprehensive Metrics** - 15+ performance indicators
- ✅ **Professional Visualizations** - Equity curves, drawdowns, trade analysis

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | > 2.0 | Excellent risk-adjusted returns |
| **Max Drawdown** | < 15% | Controlled risk exposure |
| **Win Rate** | > 55% | More winners than losers |
| **Profit Factor** | > 2.0 | Wins 2x larger than losses |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For GPU support, ensure you have CUDA installed. Check compatibility:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 2. Fetch Data (if needed)

```bash
python data/fetch_mt5.py
```

This will download 50,000 bars of XAU/USD 5M data from MetaTrader 5.

### 3. Train the Model

```bash
python train_v3.py
```

**Training Configuration:**
- Symbol: XAU/USD
- Timeframe: 5M
- Total timesteps: 500,000 (adjust as needed)
- GPU: Automatically detected
- Logs: `logs/` directory
- Models: `models/` directory

**Monitor Training:**
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006 in your browser.

### 4. Evaluate the Model

```bash
# Single evaluation
python evaluate_v3.py --model models/xauusd_m5_ppo_final.zip --data data/xauusd_m5.csv

# Cross-timeframe evaluation
python evaluate_v3.py --model models/xauusd_m5_ppo_final.zip --cross-tf
```

Results will be saved to `results/` directory with:
- Performance metrics (CSV)
- Equity curves (PNG)
- Trade distribution (PNG)
- Rolling Sharpe ratio (PNG)
- Action distribution (PNG)
- Strategy comparison (PNG)

## 📁 Project Structure

```
PPO_trader/
├── data/
│   ├── fetch_mt5.py          # Data fetching from MT5
│   ├── eurusd_m5.csv          # EUR/USD 5M data
│   └── xauusd_m5.csv          # XAU/USD 5M data
│
├── features/
│   ├── indicators.py          # Original indicators (v1)
│   └── indicators_v2.py       # Enhanced indicators (35+ features)
│
├── env/
│   ├── trading_env.py         # Original environment (v1)
│   ├── trading_env_v2.py      # Improved environment (v2)
│   └── trading_env_v3.py      # Advanced environment (v3) ⭐
│
├── evaluation/
│   ├── metrics.py             # Performance metrics
│   └── visualizations.py      # Plotting utilities
│
├── models/                    # Saved models
├── logs/                      # TensorBoard logs
├── results/                   # Evaluation results
│
├── train.py                   # Original training (v1)
├── train_v3.py                # Enhanced training (v3) ⭐
├── evaluate.py                # Original evaluation (v1)
├── evaluate_v3.py             # Enhanced evaluation (v3) ⭐
│
└── requirements.txt           # Dependencies
```

## 🔧 Technical Details

### Feature Engineering (indicators_v2.py)

**Trend Indicators:**
- RSI (14), MACD, ADX (14)
- Bollinger Bands (20, 2σ)
- SMMA (9), TEMA (14)

**Momentum Indicators:**
- Stochastic Oscillator (14, 3)
- Rate of Change (10)
- Momentum (10)

**Volatility Indicators:**
- ATR (14)
- Historical Volatility (20)
- Bollinger Band Width

**Volume Indicators:**
- Volume MA (20)
- Volume Ratio
- On-Balance Volume (OBV)

**Multi-Timeframe:**
- 15M RSI, Trend
- 30M Trend

**Time Features:**
- Hour (sin/cos encoding)
- Day of week (sin/cos encoding)
- Market session (Asian/European/US)

**Market Regime:**
- Trending vs Ranging
- High vs Low Volatility

### Trading Environment (trading_env_v3.py)

**Action Space:**
- Continuous: [-1.0, 1.0]
- -1.0 = 100% short
- 0.0 = Flat (no position)
- +1.0 = 100% long

**Observation Space:**
- 35+ technical features
- Current position
- Unrealized PnL
- Current drawdown

**Risk Management:**
- Stop-loss: 2%
- Take-profit: 4%
- Max drawdown limit: 20%
- Transaction cost: 0.02%

**Reward Function:**
```python
reward = step_return * 100
       + sharpe_bonus * 0.01
       - drawdown_penalty * 0.1
       + win_rate_bonus * 0.01
```

### Model Architecture

**PPO Hyperparameters:**
- Policy: MlpPolicy (3 layers)
- Learning rate: 3e-4
- Steps per rollout: 2048
- Batch size: 64
- Epochs per update: 10
- Entropy coefficient: 0.01 (exploration)
- Clip range: 0.2

## 📈 Results Comparison

### Original Model (v1)
- Features: 4
- Action: Discrete (Hold/Buy/Sell)
- XAU/USD 5M PnL: +$251 (2.5%)
- No risk management

### Enhanced Model (v3)
- Features: 35+
- Action: Continuous (position sizing)
- Risk management: ✅
- Target Sharpe: > 2.0
- Target Drawdown: < 15%

## 🎓 Training Tips

### For Better Performance:

1. **Increase Training Time**
   ```python
   total_timesteps=1_000_000  # Instead of 500k
   ```

2. **Adjust Entropy Coefficient**
   ```python
   ent_coef=0.02  # More exploration
   ```

3. **Tune Reward Scaling**
   ```python
   reward_scaling=200.0  # Stronger signals
   ```

4. **Use Learning Rate Schedule**
   ```python
   learning_rate=lambda progress: 3e-4 * (1 - progress)
   ```

### Hyperparameter Optimization

For automated tuning, use Optuna:
```python
# TODO: Implement hyperparameter optimization script
```

## 🔍 Monitoring Training

### TensorBoard Metrics

**Rollout:**
- `ep_len_mean` - Average episode length
- `ep_rew_mean` - Average episode reward

**Training:**
- `approx_kl` - KL divergence (should be small)
- `clip_fraction` - Fraction of clipped updates
- `entropy_loss` - Exploration level
- `explained_variance` - Value function quality
- `policy_gradient_loss` - Policy improvement
- `value_loss` - Value function error

**Custom Metrics:**
- `eval/sharpe_ratio` - Validation Sharpe
- `eval/max_drawdown` - Validation drawdown
- `eval/win_rate` - Validation win rate
- `eval/profit_factor` - Validation profit factor

### Good Training Signs:
✅ `ep_rew_mean` increasing
✅ `explained_variance` > 0.5
✅ `entropy_loss` gradually decreasing
✅ `eval/sharpe_ratio` > 1.0

### Bad Training Signs:
❌ `ep_rew_mean` negative or decreasing
❌ `explained_variance` negative
❌ `approx_kl` > 0.1 (too aggressive updates)
❌ `eval/sharpe_ratio` < 0

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size=32  # Instead of 64
```

### Training Too Slow
```python
# Reduce n_steps
n_steps=1024  # Instead of 2048
```

### Poor Performance
1. Check feature quality (NaN values)
2. Verify data quality (outliers, gaps)
3. Increase training time
4. Adjust reward function
5. Try different hyperparameters

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## ⚠️ Disclaimer

This is a research/educational project. **DO NOT use this for live trading without extensive testing and validation.** Past performance does not guarantee future results. Trading involves substantial risk of loss.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Built with ❤️ for algorithmic trading research**
