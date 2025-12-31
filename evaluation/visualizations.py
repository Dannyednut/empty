"""
Visualization utilities for trading strategy performance
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_equity_curve(
    portfolio_values: List[float],
    timestamps: Optional[List] = None,
    title: str = "Equity Curve",
    show_drawdown: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot equity curve with optional drawdown overlay
    """
    fig, axes = plt.subplots(2 if show_drawdown else 1, 1, figsize=(14, 10 if show_drawdown else 6))
    
    if not show_drawdown:
        axes = [axes]
    
    # Equity curve
    if timestamps is not None:
        axes[0].plot(timestamps, portfolio_values, linewidth=2, color='#2E86AB')
    else:
        axes[0].plot(portfolio_values, linewidth=2, color='#2E86AB')
    
    axes[0].set_title(title, fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='y')
    
    # Add horizontal line at initial value
    axes[0].axhline(y=portfolio_values[0], color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    axes[0].legend()
    
    # Drawdown
    if show_drawdown:
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        
        if timestamps is not None:
            axes[1].fill_between(timestamps, drawdown, 0, color='#A23B72', alpha=0.6)
            axes[1].plot(timestamps, drawdown, linewidth=1.5, color='#A23B72')
        else:
            axes[1].fill_between(range(len(drawdown)), drawdown, 0, color='#A23B72', alpha=0.6)
            axes[1].plot(drawdown, linewidth=1.5, color='#A23B72')
        
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].set_xlabel('Time' if timestamps is not None else 'Steps', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_trade_distribution(
    trades: List[Dict],
    title: str = "Trade Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of winning and losing trades
    """
    if len(trades) == 0:
        print("No trades to plot")
        return
    
    pnls = [trade.get('pnl', 0) * 100 for trade in trades]  # Convert to percentage
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(wins, bins=20, alpha=0.7, color='#06A77D', label=f'Wins ({len(wins)})', edgecolor='black')
    axes[0].hist(losses, bins=20, alpha=0.7, color='#D62246', label=f'Losses ({len(losses)})', edgecolor='black')
    axes[0].set_xlabel('PnL (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = []
    labels = []
    if len(wins) > 0:
        data_to_plot.append(wins)
        labels.append('Wins')
    if len(losses) > 0:
        data_to_plot.append(losses)
        labels.append('Losses')
    
    if data_to_plot:
        bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['#06A77D', '#D62246']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    axes[1].set_ylabel('PnL (%)', fontsize=12)
    axes[1].set_title('Trade PnL Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_monthly_returns(
    portfolio_values: List[float],
    timestamps: List,
    title: str = "Monthly Returns Heatmap",
    save_path: Optional[str] = None
):
    """
    Plot monthly returns as a heatmap
    """
    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'value': portfolio_values
    })
    df.set_index('timestamp', inplace=True)
    
    # Calculate monthly returns
    monthly = df.resample('M').last()
    monthly_returns = monthly.pct_change() * 100
    
    # Create pivot table for heatmap
    monthly_returns['year'] = monthly_returns.index.year
    monthly_returns['month'] = monthly_returns.index.month
    
    pivot = monthly_returns.pivot_table(values='value', index='year', columns='month', aggfunc='mean')
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Return (%)'}, ax=ax, linewidths=0.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    # Set month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rolling_sharpe(
    portfolio_values: List[float],
    window: int = 500,
    periods_per_year: int = 252 * 24 * 12,
    title: str = "Rolling Sharpe Ratio",
    save_path: Optional[str] = None
):
    """
    Plot rolling Sharpe ratio
    """
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    
    rolling_sharpe = []
    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        sharpe = np.mean(window_returns) / (np.std(window_returns) + 1e-8) * np.sqrt(periods_per_year)
        rolling_sharpe.append(sharpe)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(window, len(returns)), rolling_sharpe, linewidth=2, color='#F18F01')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
    ax.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.5, label='Sharpe = 2.0')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_action_distribution(
    actions: List[float],
    title: str = "Action Distribution Over Time",
    save_path: Optional[str] = None
):
    """
    Plot distribution of actions taken by the agent
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Action over time
    axes[0].plot(actions, linewidth=1, alpha=0.7, color='#5E60CE')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].fill_between(range(len(actions)), actions, 0, alpha=0.3, color='#5E60CE')
    axes[0].set_ylabel('Position Size', fontsize=12)
    axes[0].set_title('Position Size Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Action histogram
    axes[1].hist(actions, bins=50, color='#5E60CE', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Position Size', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Position Size Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison(
    strategies: Dict[str, List[float]],
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None
):
    """
    Compare multiple strategies on the same plot
    
    Args:
        strategies: Dict with strategy names as keys and portfolio values as values
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62246']
    
    for i, (name, values) in enumerate(strategies.items()):
        # Normalize to percentage returns
        normalized = np.array(values) / values[0] * 100
        ax.plot(normalized, label=name, linewidth=2, color=colors[i % len(colors)])
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Portfolio Value (% of Initial)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")
    
    # Generate sample data
    np.random.seed(42)
    n_steps = 1000
    
    # Portfolio values
    returns = np.random.normal(0.0005, 0.01, n_steps)
    portfolio_values = [10000]
    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    # Timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=len(portfolio_values), freq='5T')
    
    # Trades
    trades = []
    for i in range(100):
        pnl = np.random.normal(0.01, 0.03)
        trades.append({'pnl': pnl})
    
    # Actions
    actions = np.random.uniform(-1, 1, n_steps)
    
    # Plot equity curve
    plot_equity_curve(portfolio_values, timestamps, title="Test Strategy Equity Curve")
    
    # Plot trade distribution
    plot_trade_distribution(trades, title="Test Trade Distribution")
    
    # Plot rolling Sharpe
    plot_rolling_sharpe(portfolio_values, window=200, title="Test Rolling Sharpe Ratio")
    
    # Plot action distribution
    plot_action_distribution(actions, title="Test Action Distribution")
    
    print("✓ All visualizations tested successfully!")
