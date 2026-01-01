"""
Enhanced Evaluation Script with Comprehensive Analysis
"""
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import sys
import os

sys.path.append('.')

from env.trading_env_v3 import AdvancedTradingEnv
from features.indicators_v2 import build_features, get_feature_columns
from evaluation.metrics import calculate_all_metrics, print_metrics_report
from evaluation.visualizations import (
    plot_equity_curve, plot_trade_distribution, 
    plot_rolling_sharpe, plot_action_distribution, plot_comparison
)


def evaluate_model(
    model_path: str,
    data_path: str,
    symbol: str = "xauusd",
    timeframe: str = "m5",
    initial_cash: float = 10_000,
    save_plots: bool = True,
    output_dir: str = "results"
):
    """
    Comprehensive model evaluation with visualizations
    
    Args:
        model_path: Path to trained model
        data_path: Path to test data CSV
        symbol: Trading symbol
        timeframe: Timeframe
        initial_cash: Initial portfolio value
        save_plots: Whether to save plots
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("PPO TRADING AGENT - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\n🤖 Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Load and prepare data
    print(f"\n📊 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Raw data shape: {df.shape}")
    
    # Build features
    base_tf = '5min' if 'm5' in timeframe else '15min' if 'm15' in timeframe else '30min'
    df = build_features(df, base_timeframe=base_tf)
    feature_columns = get_feature_columns()
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    print(f"   Features: {len(feature_columns)}")
    print(f"   Enhanced data shape: {df.shape}")
    
    # Use last 30% as test set
    test_df = df.iloc[int(len(df) * 0.7):].reset_index(drop=True)
    print(f"   Test set size: {len(test_df)}")
    
    # Create environment
    print(f"\n🏗️  Creating test environment...")
    env = AdvancedTradingEnv(
        df=test_df,
        feature_columns=feature_columns,
        initial_cash=initial_cash,
        episode_length=len(test_df),
        transaction_cost=0.0002,
        max_position_size=1.0,
        stop_loss_pct=0.005,  # 0.5% SL
        take_profit_pct=0.015, # 1.5% TP
        max_drawdown_limit=0.50,  # Allow full evaluation without circuit breaker if possible
        reward_scaling=1.0  # Use new scaling
    )
    
    # Wrap in DummyVecEnv so we can apply normalization
    venv = DummyVecEnv([lambda: env])
    
    # Load normalization stats if they exist
    # Robustly handle .zip in model_path
    base_model_path = model_path[:-4] if model_path.endswith('.zip') else model_path
    vec_normalize_path = base_model_path + "_vec_normalize.pkl"
    
    vec_normalize = None
    if os.path.exists(vec_normalize_path):
        print(f"   📈 Loading normalization stats: {vec_normalize_path}")
        vec_normalize = VecNormalize.load(vec_normalize_path, venv)
        vec_normalize.training = False  # Don't update stats during eval
        vec_normalize.norm_reward = False  # Don't normalize rewards during eval
    else:
        print(f"   ⚠️  No normalization stats found at {vec_normalize_path}. Observations will be raw.")
    
    # Check feature consistency
    expected_n_features = model.observation_space.shape[0] - 3  # env adds 3 portfolio features
    actual_n_features = len(feature_columns)
    if expected_n_features != actual_n_features:
        print(f"\n❌ Feature mismatch! Model expects {expected_n_features} features, but got {actual_n_features}.")
        print(f"   Check your get_feature_columns() and build_features() output.")
        # But try to proceed if possible or exit
    
    # Run evaluation
    print(f"\n🚀 Running evaluation...")
    obs, _ = env.reset()
    done = False
    
    portfolio_values = []
    actions = []
    timestamps = []
    
    step = 0
    while not done:
        # Use vec_normalize to normalize observation if available
        normalized_obs = vec_normalize.normalize_obs(obs) if vec_normalize else obs
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        actions.append(action[0])
        timestamps.append(test_df.iloc[min(step, len(test_df)-1)]['time'])
        
        step += 1
        
        if step % 1000 == 0:
            print(f"   Step {step}/{len(test_df)} - Portfolio: ${info['portfolio_value']:,.2f}")
    
    print(f"\n✅ Evaluation complete!")
    
    # Get comprehensive metrics
    print(f"\n📊 Calculating metrics...")
    metrics = calculate_all_metrics(env.portfolio_values, env.trades, initial_cash)
    
    # Print metrics report
    print_metrics_report(metrics, f"{symbol.upper()} {timeframe.upper()} - Test Performance")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_file = f"{output_dir}/{symbol}_{timeframe}_test_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n💾 Metrics saved: {metrics_file}")
    
    # Generate visualizations
    if save_plots:
        print(f"\n📈 Generating visualizations...")
        
        # Equity curve
        print("   - Equity curve with drawdown...")
        plot_equity_curve(
            portfolio_values,
            timestamps=pd.to_datetime(timestamps),
            title=f"{symbol.upper()} {timeframe.upper()} - Equity Curve",
            show_drawdown=True,
            save_path=f"{output_dir}/{symbol}_{timeframe}_equity_curve.png"
        )
        
        # Trade distribution
        if len(env.trades) > 0:
            print("   - Trade distribution...")
            plot_trade_distribution(
                env.trades,
                title=f"{symbol.upper()} {timeframe.upper()} - Trade Distribution",
                save_path=f"{output_dir}/{symbol}_{timeframe}_trade_dist.png"
            )
        
        # Rolling Sharpe
        if len(portfolio_values) > 500:
            print("   - Rolling Sharpe ratio...")
            plot_rolling_sharpe(
                portfolio_values,
                window=500,
                title=f"{symbol.upper()} {timeframe.upper()} - Rolling Sharpe Ratio",
                save_path=f"{output_dir}/{symbol}_{timeframe}_rolling_sharpe.png"
            )
        
        # Action distribution
        print("   - Action distribution...")
        plot_action_distribution(
            actions,
            title=f"{symbol.upper()} {timeframe.upper()} - Position Sizing",
            save_path=f"{output_dir}/{symbol}_{timeframe}_actions.png"
        )
        
        print(f"\n✅ Visualizations saved to: {output_dir}/")
    
    # Baseline comparison: Buy and Hold
    print(f"\n📊 Baseline Comparison: Buy and Hold")
    print("-" * 80)
    
    buy_hold_values = [initial_cash]
    entry_price = test_df.iloc[0]['close']
    shares = initial_cash / entry_price
    
    for i in range(1, len(test_df)):
        current_price = test_df.iloc[i]['close']
        buy_hold_values.append(shares * current_price)
    
    buy_hold_return = (buy_hold_values[-1] - initial_cash) / initial_cash
    ppo_return = metrics['total_return']
    
    print(f"Buy & Hold Return:  {buy_hold_return:>10.2%}")
    print(f"PPO Agent Return:   {ppo_return:>10.2%}")
    print(f"Outperformance:     {(ppo_return - buy_hold_return):>10.2%}")
    
    if ppo_return > buy_hold_return:
        print(f"\n✅ PPO agent outperformed buy-and-hold by {(ppo_return - buy_hold_return)*100:.2f}%")
    else:
        print(f"\n⚠️  PPO agent underperformed buy-and-hold by {(buy_hold_return - ppo_return)*100:.2f}%")
    
    # Plot comparison
    if save_plots:
        plot_comparison(
            {
                'PPO Agent': portfolio_values,
                'Buy & Hold': buy_hold_values
            },
            title=f"{symbol.upper()} {timeframe.upper()} - Strategy Comparison",
            save_path=f"{output_dir}/{symbol}_{timeframe}_comparison.png"
        )
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return:          {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:          {metrics['max_drawdown']:.2%}")
    print(f"Win Rate:              {metrics['win_rate']:.2%}")
    print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
    print(f"Total Trades:          {metrics['total_trades']}")
    print("=" * 80)
    
    return metrics, portfolio_values, actions


def cross_timeframe_evaluation(model_path: str, symbol: str = "xauusd"):
    """
    Evaluate model across multiple timeframes
    """
    print("\n" + "=" * 80)
    print("CROSS-TIMEFRAME EVALUATION")
    print("=" * 80)
    
    timeframes = ['m5', 'm15']  # Add 'm30' if you have the data
    results = {}
    
    for tf in timeframes:
        data_path = f"data/{symbol}_{tf}.csv"
        if os.path.exists(data_path):
            print(f"\n{'='*80}")
            print(f"Evaluating on {tf.upper()} timeframe...")
            print(f"{'='*80}")
            
            metrics, _, _ = evaluate_model(
                model_path=model_path,
                data_path=data_path,
                symbol=symbol,
                timeframe=tf,
                save_plots=True,
                output_dir=f"results/{tf}"
            )
            
            results[tf] = metrics
        else:
            print(f"\n⚠️  Data file not found: {data_path}")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("CROSS-TIMEFRAME SUMMARY")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(results).T
        print(comparison_df[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']])
        
        comparison_df.to_csv("results/cross_timeframe_comparison.csv")
        print(f"\n💾 Comparison saved: results/cross_timeframe_comparison.csv")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PPO trading agent')
    parser.add_argument('--model', type=str, default='models/xauusd/experts/xauusd_m5_ppo_expert',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/xauusd_m5.csv',
                        help='Path to test data')
    parser.add_argument('--symbol', type=str, default='xauusd',
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='m5',
                        help='Timeframe (m5, m15, m30)')
    parser.add_argument('--cross-tf', action='store_true',
                        help='Run cross-timeframe evaluation')
    
    args = parser.parse_args()
    
    if args.cross_tf:
        # Cross-timeframe evaluation
        cross_timeframe_evaluation(args.model, args.symbol)
    else:
        # Single evaluation
        evaluate_model(
            model_path=args.model,
            data_path=args.data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            save_plots=True,
            output_dir="results"
        )
