import numpy as np

def build_features(df, sma=9, tema=14):
    close = df["close"]

    # SMMA (simple MA approximation)
    smma = close.rolling(sma).mean()

    # TEMA
    ema1 = close.ewm(span=tema, adjust=False).mean()
    ema2 = ema1.ewm(span=tema, adjust=False).mean()
    ema3 = ema2.ewm(span=tema, adjust=False).mean()
    tema_val = 3 * (ema1 - ema2) + ema3

    df["diff"] = smma - tema_val
    df["diff_prev"] = df["diff"].shift(1)
    df["candle_range"] = df["high"] - df["low"]

    df = df.dropna().reset_index(drop=True)

    # Normalize (critical)
    for col in ["diff", "diff_prev", "candle_range"]:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    return df
