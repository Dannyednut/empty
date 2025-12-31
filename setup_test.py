"""
Quick Setup and Test Script
Verifies installation and tests all components
"""
import sys
import subprocess


def check_dependencies():
    """Check if all dependencies are installed"""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    dependencies = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'gymnasium': 'RL environment framework',
        'stable_baselines3': 'PPO implementation',
        'torch': 'Deep learning (GPU support)',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'ta': 'Technical analysis (optional)'
    }
    
    missing = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"{package:<20} - {description}")
        except ImportError:
            print(f"{package:<20} - {description} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    else:
        print(f"\nAll dependencies installed!")
        return True


def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 80)
    print("CHECKING GPU")
    print("=" * 80)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA Available: YES")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"CUDA Available: NO")
            print(f"   Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False


def test_features():
    """Test feature engineering"""
    print("\n" + "=" * 80)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 80)
    
    try:
        import pandas as pd
        from features.indicators_v2 import build_features, get_feature_columns
        
        # Check if data exists
        import os
        if not os.path.exists("data/xauusd_m5.csv"):
            print("XAU/USD data not found. Run: python data/fetch_mt5.py")
            return False
        
        # Load sample data
        df = pd.read_csv("data/xauusd_m5.csv")
        print(f"   Loaded data: {df.shape}")
        
        # Build features
        df_features = build_features(df.head(1000), base_timeframe='5min')
        feature_cols = get_feature_columns()
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        print(f"   Features created: {len(feature_cols)}")
        print(f"   Enhanced data: {df_features.shape}")
        
        # Check for NaN
        nan_count = df_features[feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"    Warning: {nan_count} NaN values found")
        else:
            print(f"    No NaN values")
        
        print(f"\nFeature engineering working correctly!")
        return True
        
    except Exception as e:
        print(f"Error testing features: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test trading environment"""
    print("\n" + "=" * 80)
    print("TESTING TRADING ENVIRONMENT")
    print("=" * 80)
    
    try:
        import pandas as pd
        import numpy as np
        from env.trading_env_v3 import AdvancedTradingEnv
        from features.indicators_v2 import build_features, get_feature_columns
        
        # Load and prepare data
        df = pd.read_csv("data/xauusd_m5.csv")
        df = build_features(df.head(1000), base_timeframe='5min')
        feature_cols = get_feature_columns()
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Create environment
        env = AdvancedTradingEnv(
            df=df,
            feature_columns=feature_cols,
            initial_cash=10_000,
            episode_length=500
        )
        
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset()
        print(f"   Reset successful: obs shape = {obs.shape}")
        
        # Test random actions
        done = False
        steps = 0
        while not done and steps < 10:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        print(f"   Ran {steps} steps successfully")
        print(f"   Final portfolio: ${info['portfolio_value']:,.2f}")
        
        print(f"\nTrading environment working correctly!")
        return True
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation"""
    print("\n" + "=" * 80)
    print("TESTING METRICS")
    print("=" * 80)
    
    try:
        import numpy as np
        from evaluation.metrics import calculate_all_metrics, print_metrics_report
        
        # Simulate portfolio values
        np.random.seed(42)
        initial_cash = 10000
        returns = np.random.normal(0.001, 0.01, 1000)
        portfolio_values = [initial_cash]
        
        for r in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + r))
        
        # Simulate trades
        trades = []
        for i in range(50):
            pnl = np.random.normal(0.01, 0.03)
            trades.append({'pnl': pnl})
        
        # Calculate metrics
        metrics = calculate_all_metrics(portfolio_values, trades, initial_cash)
        
        print(f"   Calculated {len(metrics)} metrics")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        
        print(f"\nMetrics calculation working correctly!")
        return True
        
    except Exception as e:
        print(f"Error testing metrics: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PPO TRADING AGENT - SETUP VERIFICATION")
    print("=" * 80)
    
    results = {}
    
    # Check dependencies
    results['dependencies'] = check_dependencies()
    
    # Check GPU
    results['gpu'] = check_gpu()
    
    # Test components (only if dependencies are installed)
    if results['dependencies']:
        results['features'] = test_features()
        results['environment'] = test_environment()
        results['metrics'] = test_metrics()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for component, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"{component.capitalize():<20} {status_str}")
    
    if all(results.values()):
        print("\nAll tests passed! You're ready to train!")
        print("\nNext steps:")
        print("1. Ensure you have data: python data/fetch_mt5.py")
        print("2. Train model: python train_v3.py")
        print("3. Evaluate: python evaluate_v3.py")
    else:
        print("\nSome tests failed. Please fix the issues above.")
        if not results['dependencies']:
            print("\nInstall dependencies: pip install -r requirements.txt")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
