import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
BARS = 50_000

def fetch_data():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        BARS
    )

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df[["time", "open", "high", "low", "close", "tick_volume"]]

if __name__ == "__main__":
    df = fetch_data()
    df.to_csv(f"data/{SYMBOL.lower()}_m15.csv", index=False)
    print("Data saved.")
