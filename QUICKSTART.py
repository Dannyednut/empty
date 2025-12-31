"""
Quick Start Guide - Train Your PPO Trading Agent
Run this after installing dependencies
"""

print("=" * 80)
print("PPO TRADING AGENT - QUICK START GUIDE")
print("=" * 80)

print("""
STEP 1: Install Dependencies
----------------------------
Run this command to install all required packages:

    pip install pandas numpy gymnasium stable-baselines3 torch matplotlib seaborn tensorboard

Or use the requirements file:

    pip install -r requirements.txt


STEP 2: Verify Setup
----------------------------
Test that everything is working:

    python setup_test.py

Expected output: All tests should PASS


STEP 3: Start Training
----------------------------
Train the PPO agent on XAU/USD 5M data:

    python train_v3.py

This will:
- Load XAU/USD 5M data (50,000 bars)
- Build 35+ technical features
- Train for 500,000 timesteps (~30-60 minutes on GPU)
- Save models to models/ directory
- Log metrics to logs/ directory

Monitor training in real-time:

    tensorboard --logdir logs/

Then open: http://localhost:6006


STEP 4: Evaluate Results
----------------------------
After training completes, evaluate the model:

    python evaluate_v3.py --model models/xauusd_m5_ppo_final.zip

This will:
- Test on unseen data (30% holdout)
- Calculate 15+ performance metrics
- Generate visualizations
- Compare against buy-and-hold baseline
- Save results to results/ directory


STEP 5: Review Performance
----------------------------
Check if your model meets success criteria:

✓ Sharpe Ratio > 2.0
✓ Max Drawdown < 15%
✓ Win Rate > 55%
✓ Profit Factor > 2.0

If not, try:
- Increase training time: total_timesteps=1_000_000
- Adjust hyperparameters (see train_v3.py)
- Add more data


TROUBLESHOOTING
----------------------------
Issue: "No module named 'pandas'"
Fix: pip install pandas numpy gymnasium stable-baselines3 torch

Issue: "CUDA out of memory"
Fix: Reduce batch_size=32 in train_v3.py

Issue: "Data file not found"
Fix: Run python data/fetch_mt5.py to download data

Issue: Training too slow
Fix: Reduce n_steps=1024 or use GPU

Issue: Poor performance
Fix: Increase total_timesteps or tune hyperparameters


NEXT STEPS
----------------------------
1. Run setup_test.py to verify installation
2. Run train_v3.py to train your first model
3. Monitor progress with TensorBoard
4. Evaluate with evaluate_v3.py
5. Iterate and improve!

Good luck! 🚀
""")

print("=" * 80)
print("Ready to start? Run: python setup_test.py")
print("=" * 80)
