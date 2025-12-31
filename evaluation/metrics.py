"""
Comprehensive Performance Metrics for Trading Strategies
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """Calculate returns from portfolio values"""
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns


def calculate_total_return(initial_value: float, final_value: float) -> float:
    """Calculate total return percentage"""
    return (final_value - initial_value) / initial_value


def calculate_annualized_return(total_return: float, n_periods: int, periods_per_year: int = 252 * 24 * 12) -> float:
    """
    Calculate annualized return
    
    Args:
        total_return: Total return as decimal (e.g., 0.15 for 15%)
        n_periods: Number of periods in the backtest
        periods_per_year: Number of periods per year (default for 5min data: 252 days * 24 hours * 12 periods)
    """
    if n_periods == 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252 * 24 * 12) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252 * 24 * 12) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
    return sortino


def calculate_max_drawdown(portfolio_values: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown
    
    Returns:
        max_dd: Maximum drawdown as decimal
        start_idx: Index where drawdown started
        end_idx: Index where drawdown bottomed
    """
    values = np.array(portfolio_values)
    peak = values[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_start = 0
    max_dd_end = 0
    
    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_idx = i
        
        dd = (value - peak) / peak
        if dd < max_dd:
            max_dd = dd
            max_dd_start = peak_idx
            max_dd_end = i
    
    return max_dd, max_dd_start, max_dd_end


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar Ratio (annualized return / max drawdown)
    """
    if max_drawdown == 0:
        return 0.0
    return annualized_return / abs(max_drawdown)


def calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from list of trades"""
    if len(trades) == 0:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    """
    if len(trades) == 0:
        return 0.0
    
    gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_average_win_loss(trades: List[Dict]) -> Tuple[float, float]:
    """
    Calculate average win and average loss
    
    Returns:
        avg_win: Average winning trade
        avg_loss: Average losing trade
    """
    if len(trades) == 0:
        return 0.0, 0.0
    
    wins = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0]
    losses = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    return avg_win, avg_loss


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    Calculate expectancy (average profit per trade)
    """
    if len(trades) == 0:
        return 0.0
    
    win_rate = calculate_win_rate(trades)
    avg_win, avg_loss = calculate_average_win_loss(trades)
    
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    return expectancy


def calculate_recovery_factor(total_return: float, max_drawdown: float) -> float:
    """
    Calculate recovery factor (total return / max drawdown)
    """
    if max_drawdown == 0:
        return 0.0
    return total_return / abs(max_drawdown)


def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252 * 24 * 12) -> float:
    """Calculate annualized volatility"""
    if len(returns) == 0:
        return 0.0
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_all_metrics(
    portfolio_values: List[float],
    trades: List[Dict],
    initial_cash: float,
    periods_per_year: int = 252 * 24 * 12,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate all performance metrics
    
    Args:
        portfolio_values: List of portfolio values over time
        trades: List of trade dictionaries with 'pnl' key
        initial_cash: Initial portfolio value
        periods_per_year: Number of periods per year (default for 5min data)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all metrics
    """
    if len(portfolio_values) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'recovery_factor': 0.0,
            'total_trades': 0,
            'final_value': initial_cash
        }
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Basic metrics
    final_value = portfolio_values[-1]
    total_return = calculate_total_return(initial_cash, final_value)
    annualized_return = calculate_annualized_return(total_return, len(portfolio_values), periods_per_year)
    
    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(portfolio_values)
    volatility = calculate_volatility(returns, periods_per_year)
    
    # Ratio metrics
    calmar = calculate_calmar_ratio(annualized_return, max_dd)
    recovery = calculate_recovery_factor(total_return, max_dd)
    
    # Trade metrics
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    avg_win, avg_loss = calculate_average_win_loss(trades)
    expectancy = calculate_expectancy(trades)
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'volatility': volatility,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'recovery_factor': recovery,
        'total_trades': len(trades)
    }


def print_metrics_report(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """Print formatted metrics report"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    print(f"\n{'Portfolio Performance':^60}")
    print("-" * 60)
    print(f"Final Value:          ${metrics.get('final_value', 0):>15,.2f}")
    print(f"Total Return:         {metrics.get('total_return', 0):>15.2%}")
    print(f"Annualized Return:    {metrics.get('annualized_return', 0):>15.2%}")
    
    print(f"\n{'Risk Metrics':^60}")
    print("-" * 60)
    print(f"Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>15.2f}")
    print(f"Sortino Ratio:        {metrics.get('sortino_ratio', 0):>15.2f}")
    print(f"Max Drawdown:         {metrics.get('max_drawdown', 0):>15.2%}")
    print(f"Volatility (Annual):  {metrics.get('volatility', 0):>15.2%}")
    print(f"Calmar Ratio:         {metrics.get('calmar_ratio', 0):>15.2f}")
    print(f"Recovery Factor:      {metrics.get('recovery_factor', 0):>15.2f}")
    
    print(f"\n{'Trade Statistics':^60}")
    print("-" * 60)
    print(f"Total Trades:         {metrics.get('total_trades', 0):>15}")
    print(f"Win Rate:             {metrics.get('win_rate', 0):>15.2%}")
    print(f"Profit Factor:        {metrics.get('profit_factor', 0):>15.2f}")
    print(f"Average Win:          {metrics.get('avg_win', 0):>15.2%}")
    print(f"Average Loss:         {metrics.get('avg_loss', 0):>15.2%}")
    print(f"Expectancy:           {metrics.get('expectancy', 0):>15.2%}")
    
    print("\n" + "=" * 60)
    
    # Performance assessment
    print(f"\n{'Performance Assessment':^60}")
    print("-" * 60)
    
    assessments = []
    if metrics['sharpe_ratio'] > 2.0:
        assessments.append("✓ Excellent Sharpe Ratio (>2.0)")
    elif metrics['sharpe_ratio'] > 1.0:
        assessments.append("✓ Good Sharpe Ratio (>1.0)")
    else:
        assessments.append("✗ Poor Sharpe Ratio (<1.0)")
    
    if abs(metrics['max_drawdown']) < 0.15:
        assessments.append("✓ Low Drawdown (<15%)")
    elif abs(metrics['max_drawdown']) < 0.25:
        assessments.append("~ Moderate Drawdown (15-25%)")
    else:
        assessments.append("✗ High Drawdown (>25%)")
    
    if metrics['win_rate'] > 0.55:
        assessments.append("✓ Good Win Rate (>55%)")
    elif metrics['win_rate'] > 0.45:
        assessments.append("~ Moderate Win Rate (45-55%)")
    else:
        assessments.append("✗ Low Win Rate (<45%)")
    
    if metrics['profit_factor'] > 2.0:
        assessments.append("✓ Excellent Profit Factor (>2.0)")
    elif metrics['profit_factor'] > 1.5:
        assessments.append("✓ Good Profit Factor (>1.5)")
    else:
        assessments.append("✗ Poor Profit Factor (<1.5)")
    
    for assessment in assessments:
        print(assessment)
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics calculation...")
    
    # Simulate portfolio values
    np.random.seed(42)
    initial_cash = 10000
    returns = np.random.normal(0.0001, 0.01, 1000)  # Simulate returns
    portfolio_values = [initial_cash]
    
    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    # Simulate trades
    trades = []
    for i in range(50):
        pnl = np.random.normal(0.01, 0.05)
        trades.append({'pnl': pnl})
    
    # Calculate metrics
    metrics = calculate_all_metrics(portfolio_values, trades, initial_cash)
    
    # Print report
    print_metrics_report(metrics, "Test Strategy Performance")
