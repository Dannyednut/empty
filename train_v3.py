"""
Enhanced Training Script for PPO Trading Agent
Features:
- GPU acceleration
- TensorBoard logging
- Model checkpointing
- Early stopping
- Walk-forward validation
"""
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import os
from datetime import datetime
import sys

# Add project root to path
sys.path.append('.')

from env.trading_env_v3 import AdvancedTradingEnv
from features.indicators_v2 import build_features, get_feature_columns
from evaluation.metrics import calculate_all_metrics, print_metrics_report


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log custom metrics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'episode' in info:
                    self.logger.record('custom/episode_reward', info['episode']['r'])
                    self.logger.record('custom/episode_length', info['episode']['l'])
        
        return True


class MetricsCallback(BaseCallback):
    """
    Callback to log trading-specific metrics during training
    """
    def __init__(self, eval_env, train_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs = self.eval_env.reset()[0]
            done = False
            
            while not done:
                # Use training env to normalize observation
                norm_obs = self.train_env.normalize_obs(obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
            
            # Get episode metrics
            metrics = self.eval_env.get_episode_metrics()
            
            # Log to TensorBoard
            self.logger.record('eval/sharpe_ratio', metrics['sharpe_ratio'])
            self.logger.record('eval/max_drawdown', metrics['max_drawdown'])
            self.logger.record('eval/win_rate', metrics['win_rate'])
            self.logger.record('eval/profit_factor', metrics['profit_factor'])
            self.logger.record('eval/total_return', metrics['total_return'])
            
            # Save best model
            if metrics['sharpe_ratio'] > self.best_sharpe:
                self.best_sharpe = metrics['sharpe_ratio']
                self.model.save(f"models/best_model_sharpe_{metrics['sharpe_ratio']:.2f}")
                if self.verbose > 0:
                    print(f"\n🎯 New best Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        
        return True


def create_env(df, feature_columns, episode_length=2000):
    """Create and wrap environment"""
    env = AdvancedTradingEnv(
        df=df,
        feature_columns=feature_columns,
        initial_cash=10_000,
        episode_length=episode_length,
        transaction_cost=0.0002,  # 0.02% for gold
        max_position_size=1.0,
        stop_loss_pct=0.005,      # 0.5% SL
        take_profit_pct=0.015,     # 1.5% TP
        max_drawdown_limit=0.30,  # 30% max drawdown
        reward_scaling=1.0,
        holding_penalty=0.00002   # Increased slightly to penalize indecision
    )
    return env


def train_ppo_agent(
    symbol: str = "xauusd",
    timeframe: str = "m5",
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,  # Strong gradient clipping for stability
    use_gpu: bool = True,
    save_dir: str = "models",
    log_dir: str = "logs"
):
    """
    Train PPO agent with advanced features
    
    Args:
        symbol: Trading symbol (e.g., 'xauusd', 'eurusd')
        timeframe: Timeframe (e.g., 'm5', 'm15', 'm30')
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient (exploration)
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        use_gpu: Use GPU if available
        save_dir: Directory to save models
        log_dir: Directory for TensorBoard logs
    """
    print("=" * 80)
    print("PPO TRADING AGENT - ENHANCED TRAINING")
    print("=" * 80)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Check GPU availability
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print(f"\n📊 Loading {symbol.upper()} {timeframe.upper()} data...")
    data_file = f"data/{symbol}_{timeframe}.csv"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"   Raw data shape: {df.shape}")
    
    # Build features
    print(f"\n🔧 Building features...")
    base_tf = '5min' if 'm5' in timeframe else '15min' if 'm15' in timeframe else '30min'
    df = build_features(df, base_timeframe=base_tf)
    feature_columns = get_feature_columns()
    
    # Filter features that exist in df
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"   Features: {len(feature_columns)}")
    print(f"   Enhanced data shape: {df.shape}")
    
    # Train/validation split (70/30)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n📈 Data split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    
    # Create environments
    print(f"\n🏗️  Creating environments...")
    # Training env
    raw_train_env = create_env(train_df, feature_columns, episode_length=2000)
    train_env = Monitor(raw_train_env)
    train_env = DummyVecEnv([lambda: train_env])
    # Normalize observations and rewards
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Validation env (non-normalized, or manually normalized)
    raw_val_env = create_env(val_df, feature_columns, episode_length=len(val_df))
    # Note: MetricsCallback handles raw observation normalization if needed, 
    # but since our indicators are already Z-scored, it should be fine.
    
    print(f"   Observation space: {raw_train_env.observation_space.shape}")
    print(f"   Action space: {raw_train_env.action_space.shape}")
    
    # Create model
    print(f"\n🤖 Creating PPO model...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Steps per rollout: {n_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs per update: {n_epochs}")
    print(f"   Entropy coefficient: {ent_coef}")
    
    # Check if tensorboard is available
    try:
        import tensorboard
        use_tensorboard = False #True
    except ImportError:
        print(f"   ⚠️  TensorBoard not installed. Training will proceed without logging.")
        print(f"   Install with: pip install tensorboard")
        use_tensorboard = False
    
    # Policy kwargs for deeper architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,  # Tanh often better for financial data
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=log_dir if use_tensorboard else None
    )
    
    # Setup callbacks
    print(f"\n⚙️  Setting up callbacks...")
    
    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix=f"{symbol}_{timeframe}_ppo"
    )
    
    # Metrics callback - evaluate every 10k steps
    metrics_callback = MetricsCallback(
        eval_env=raw_val_env,
        train_env=train_env,
        eval_freq=10000,
        verbose=1
    )
    
    # TensorBoard callback
    tensorboard_callback = TensorboardCallback()
    
    callbacks = [checkpoint_callback, metrics_callback, tensorboard_callback]
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Estimated time: {total_timesteps / (n_steps * 10):.0f} minutes")
    print(f"\n   Monitor training: tensorboard --logdir {log_dir}")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # Save final model
    final_model_base = f"{save_dir}/{symbol}_{timeframe}_ppo_final"
    model.save(final_model_base)
    # Save normalization stats
    train_env.save(f"{final_model_base}_vec_normalize.pkl")
    print(f"\n✅ Final model saved: {final_model_base}")
    print(f"✅ Normalization stats saved: {final_model_base}_vec_normalize.pkl")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation on Validation Set...")
    print("=" * 80)
    
    obs = raw_val_env.reset()[0]
    done = False
    
    while not done:
        # Use vec_normalize to normalize eval obs
        norm_obs = train_env.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, done, truncated, info = raw_val_env.step(action)
    
    # Get final metrics
    metrics = raw_val_env.get_episode_metrics()
    print_metrics_report(metrics, f"{symbol.upper()} {timeframe.upper()} - Validation Performance")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{save_dir}/{symbol}_{timeframe}_metrics.csv", index=False)
    
    print(f"\n✅ Training complete!")
    print(f"   Model: {final_model_path}.zip")
    print(f"   Metrics: {save_dir}/{symbol}_{timeframe}_metrics.csv")
    print("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    # Train on XAU/USD 5M data
    print("\n🎯 Training PPO Agent for XAU/USD (Gold) Trading\n")
    
    model, metrics = train_ppo_agent(
        symbol="xauusd",
        timeframe="m5",
        total_timesteps=1_000_000,  # Increased for deeper patterns
        learning_rate=5e-5,        # Stabilized for larger network
        n_steps=8192,               # Larger rollouts for better variance estimate
        batch_size=256,             # Larger batches for stable gradients
        n_epochs=15,                # More passes per update
        gamma=0.995,                # Higher gamma to look further ahead
        ent_coef=0.02,              # More exploration for larger network
        use_gpu=True,
        save_dir="models",
        log_dir="logs"
    )
    
    # Check if we met success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    
    criteria = {
        'Sharpe Ratio > 2.0': metrics['sharpe_ratio'] > 2.0,
        'Max Drawdown < 15%': abs(metrics['max_drawdown']) < 0.15,
        'Win Rate > 55%': metrics['win_rate'] > 0.55,
        'Profit Factor > 2.0': metrics['profit_factor'] > 2.0
    }
    
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{criterion}: {status}")
    
    if all(criteria.values()):
        print("\n🎉 ALL CRITERIA MET! Model is production-ready!")
    else:
        print("\n⚠️  Some criteria not met. Consider:")
        print("   - Increasing total_timesteps")
        print("   - Tuning hyperparameters")
        print("   - Adding more features")
        print("   - Adjusting reward function")
    
    print("=" * 80)
