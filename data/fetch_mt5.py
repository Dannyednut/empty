import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 1_000_000

def fetch_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M5, bars=1_000_000, days=None):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    if type(timeframe) == str:
        timeframe = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}")
    if not days:
        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe,
            0,
            bars,
        )
    else:
        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            datetime.now() - timedelta(days=days),
            datetime.now(),
        )

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df[["time", "open", "high", "low", "close", "tick_volume"]]

if __name__ == "__main__":
    df = fetch_data(bars=5_000_000)
    df.to_csv(f"data/{SYMBOL.lower()}_m5.csv", index=False)
    print("Data saved.")
