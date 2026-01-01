"""
Advanced Technical Indicators for Trading
Optimized for XAU/USD (Gold) with multi-timeframe analysis
"""
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength)"""
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def resample_to_higher_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample data to higher timeframe
    timeframe: '5min' (5min), '15min' (15min), '30min' (30min), '1H', etc.
    """
    df_copy = df.copy()
    df_copy['time'] = pd.to_datetime(df_copy['time'])
    df_copy.set_index('time', inplace=True)
    
    resampled = df_copy.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-based features"""
    df['time'] = pd.to_datetime(df['time'])
    
    # Hour of day (0-23) - cyclical encoding
    df['hour'] = df['time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0-6) - cyclical encoding
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Market session (simplified)
    # Asian: 0-8, European: 8-16, US: 16-24 (UTC)
    df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(float)
    df['session_european'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(float)
    df['session_us'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(float)
    
    return df


def detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Detect market regime: trending vs ranging
    Uses ADX and volatility
    """
    adx = calculate_adx(df['high'], df['low'], df['close'], period=14)
    
    # Trending: ADX > 25
    # Ranging: ADX < 20
    df['regime_trending'] = (adx > 25).astype(float)
    df['regime_ranging'] = (adx < 20).astype(float)
    
    # Volatility regime (high/low)
    volatility = df['close'].pct_change().rolling(window=lookback).std()
    vol_median = volatility.rolling(window=lookback*2).median()
    df['regime_high_vol'] = (volatility > vol_median * 1.5).astype(float)
    df['regime_low_vol'] = (volatility < vol_median * 0.5).astype(float)
    
    return df


def build_features(df: pd.DataFrame, base_timeframe: str = '5min') -> pd.DataFrame:
    """
    Build comprehensive feature set for trading
    Optimized for XAU/USD with multi-timeframe analysis
    
    Args:
        df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        base_timeframe: Base timeframe string ('5min' for 5min, '15min' for 15min, etc.)
    
    Returns:
        DataFrame with all features added
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['tick_volume']
    
    # ============================================
    # 1. ORIGINAL FEATURES (keep for compatibility)
    # ============================================
    sma = close.rolling(9).mean()
    ema1 = close.ewm(span=14, adjust=False).mean()
    ema2 = ema1.ewm(span=14, adjust=False).mean()
    ema3 = ema2.ewm(span=14, adjust=False).mean()
    tema = 3 * (ema1 - ema2) + ema3
    
    df['diff'] = sma - tema
    df['diff_prev'] = df['diff'].shift(1)
    df['candle_range'] = high - low
    
    # ============================================
    # 2. TREND INDICATORS
    # ============================================
    df['rsi'] = calculate_rsi(close, period=14)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    
    macd, signal, histogram = calculate_macd(close)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    df['macd_cross'] = np.sign(histogram)  # +1 bullish, -1 bearish
    
    df['adx'] = calculate_adx(high, low, close, period=14)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # Normalized width
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)  # 0-1 position in bands
    
    # ============================================
    # 3. MOMENTUM INDICATORS
    # ============================================
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    df['roc'] = close.pct_change(periods=10) * 100  # Rate of change
    df['momentum'] = close - close.shift(10)
    
    # ============================================
    # 4. VOLATILITY INDICATORS
    # ============================================
    df['atr'] = calculate_atr(high, low, close, period=14)
    df['atr_pct'] = df['atr'] / close  # ATR as % of price
    df['volatility'] = close.pct_change().rolling(window=20).std()
    
    # ============================================
    # 5. VOLUME INDICATORS
    # ============================================
    df['volume_ma'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / (df['volume_ma'] + 1e-10)
    df['obv'] = calculate_obv(close, volume)
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    
    # ============================================
    # 6. PRICE ACTION FEATURES
    # ============================================
    df['high_20'] = high.rolling(window=20).max()
    df['low_20'] = low.rolling(window=20).min()
    df['dist_from_high'] = (df['high_20'] - close) / close
    df['dist_from_low'] = (close - df['low_20']) / close
    
    # Candle patterns (simplified)
    body = abs(close - df['open'])
    range_candle = high - low
    df['candle_body_ratio'] = body / (range_candle + 1e-10)
    df['is_doji'] = (df['candle_body_ratio'] < 0.1).astype(float)
    
    # ============================================
    # 7. MULTI-TIMEFRAME FEATURES (Robust Implementation)
    # ============================================
    # Ensure columns exist for all base timeframes to prevent model mismatch
    
    if base_timeframe == '5min':
        # Standard: Resample 5m to 15m and 30m
        df_15m = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']], '15min')
        df_15m['rsi_15m'] = calculate_rsi(df_15m['close'], period=14)
        df_15m['ema_15m'] = df_15m['close'].ewm(span=20, adjust=False).mean()
        df_15m['trend_15m'] = (df_15m['close'] > df_15m['ema_15m']).astype(float)
        
        df = pd.merge_asof(
            df.sort_values('time'),
            df_15m[['time', 'rsi_15m', 'trend_15m']].sort_values('time'),
            on='time', direction='backward'
        )
        
        df_30m = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']], '30min')
        df_30m['ema_30m'] = df_30m['close'].ewm(span=20, adjust=False).mean()
        df_30m['trend_30m'] = (df_30m['close'] > df_30m['ema_30m']).astype(float)
        
        df = pd.merge_asof(
            df.sort_values('time'),
            df_30m[['time', 'trend_30m']].sort_values('time'),
            on='time', direction='backward'
        )
    
    elif base_timeframe == '15min':
        # Data is already 15m
        df['rsi_15m'] = df['rsi']
        ema_15m = df['close'].ewm(span=20, adjust=False).mean()
        df['trend_15m'] = (df['close'] > ema_15m).astype(float)
        
        # Resample 15m to 30m
        df_30m = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']], '30min')
        df_30m['ema_30m'] = df_30m['close'].ewm(span=20, adjust=False).mean()
        df_30m['trend_30m'] = (df_30m['close'] > df_30m['ema_30m']).astype(float)
        
        df = pd.merge_asof(
            df.sort_values('time'),
            df_30m[['time', 'trend_30m']].sort_values('time'),
            on='time', direction='backward'
        )
        
    elif base_timeframe == '30min':
        # Data is already 30m
        df['rsi_15m'] = df['rsi']
        ema_30m = df['close'].ewm(span=20, adjust=False).mean()
        df['trend_15m'] = (df['close'] > ema_30m).astype(float)
        df['trend_30m'] = df['trend_15m']
    
    else:
        # Fallback for other timeframes
        df['rsi_15m'] = df['rsi']
        df['trend_15m'] = 0.0
        df['trend_30m'] = 0.0
    
    # ============================================
    # 8. TIME-BASED FEATURES
    # ============================================
    df = add_time_features(df)
    
    # ============================================
    # 9. MARKET REGIME DETECTION
    # ============================================
    df = detect_market_regime(df)
    
    # ============================================
    # 10. DROP NaN and NORMALIZE
    # ============================================
    df = df.dropna().reset_index(drop=True)
    
    # List of features to normalize (exclude binary/categorical features)
    features_to_normalize = [
        'diff', 'diff_prev', 'candle_range',
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx',
        'bb_width', 'bb_position',
        'stoch_k', 'stoch_d', 'roc', 'momentum',
        'atr', 'atr_pct', 'volatility',
        'volume_ratio', 'obv', 'obv_ma',
        'dist_from_high', 'dist_from_low', 'candle_body_ratio'
    ]
    
    # Add multi-timeframe features if they exist
    if 'rsi_15m' in df.columns:
        features_to_normalize.append('rsi_15m')
    
    # Z-score normalization (critical for stable learning)
    for col in features_to_normalize:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)
    
    return df


def get_feature_columns() -> list:
    """
    Returns list of feature column names for observation space
    """
    base_features = [
        # Original features
        'diff', 'diff_prev', 'candle_range',
        
        # Trend indicators
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'macd_histogram', 'macd_cross', 'adx',
        'bb_width', 'bb_position',
        
        # Momentum
        'stoch_k', 'stoch_d', 'roc',
        
        # Volatility
        'atr_pct', 'volatility',
        
        # Volume
        'volume_ratio',
        
        # Price action
        'dist_from_high', 'dist_from_low', 'candle_body_ratio', 'is_doji',
        
        # Time features
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'session_asian', 'session_european', 'session_us',
        
        # Market regime
        'regime_trending', 'regime_ranging', 'regime_high_vol', 'regime_low_vol',
        
        # Multi-timeframe (if available)
        'rsi_15m', 'trend_15m', 'trend_30m'
    ]
    
    return base_features


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing feature engineering...")
    
    # Load sample data
    df = pd.read_csv("data/xauusd_m5.csv")
    print(f"Original data shape: {df.shape}")
    
    # Build features
    df_features = build_features(df, base_timeframe='5min')
    print(f"Enhanced data shape: {df_features.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Check for missing features
    missing = [col for col in feature_cols if col not in df_features.columns]
    if missing:
        print(f"\nWarning: Missing features: {missing}")
    else:
        print("\n✓ All features present!")
    
    # Show sample
    print(f"\nSample features:\n{df_features[feature_cols].head()}")
