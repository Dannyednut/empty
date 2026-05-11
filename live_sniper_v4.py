"""
PPO Sniper V4 - Live MT5 Adaptive Scalper
Features:
- Dual-Head Inference (Direction + Multiplier)
- Dynamic ATR-based Stop Loss & Take Profit
- Real-time Memory Parity with Training Environment
- Sortino-optimized Execution Logic
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import time
import os
import sys
from datetime import datetime

# Project imports
sys.path.append('.')
from features.indicators_v2 import build_features, get_feature_columns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
import argparse

# Parse command line arguments to override defaults
parser = argparse.ArgumentParser(description="PPO Trading Agent - Live MT5 Execution Bridge")
parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (e.g., XAUUSD)")
parser.add_argument("--tf", type=str, default="M5", choices=["M1", "M5", "M15", "M30", "H1"], help="Timeframe")
parser.add_argument("--lots", type=float, default=0.01, help="Lot size (fixed)")
parser.add_argument("--dry_run", type=bool, default=False, help="Dry run mode (True/False)")
args = parser.parse_args()

SYMBOL = args.symbol
DRY_RUN = args.dry_run
TIMEFRAME = getattr(mt5, f"TIMEFRAME_{args.tf.upper()}")
BASE_TIMEFRAME_STR = '5min' if args.tf.upper() == 'M5' else '15min' if args.tf.upper() == 'M15' else '30min'
MODEL_PATH = f"models/{args.symbol.lower()}/sniper/{args.symbol.lower()}_{args.tf.lower()}_sniper_v4_expert.zip"
STATS_PATH = f"models/{args.symbol.lower()}/sniper/{args.symbol.lower()}_{args.tf.lower()}_sniper_v4_vec_normalize.pkl"

# Trading Params
MAGIC_NUMBER = 20260103
LOT_SIZE = args.lots
REWARD_TO_RISK = 2.5
NEUTRAL_THRESHOLD = 0.15
LOG_FILE = "logs/sniper_v4_live.csv"

class SniperLiveV4:
    def __init__(self):
        self.model = None
        self.vec_normalize = None
        self.feature_columns = get_feature_columns()
        self.last_candle_time = None
        self.filling_mode = None
        
        # State Parity
        self.last_pos = 0.0
        self.last_mult = 1.0
        self.peak_equity = 0.0
        
        os.makedirs("logs", exist_ok=True)
        self._init_log()
        self._init_mt5()
        self._load_brain()

    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            headers = "timestamp,candle_time,equity,pos_act,mult_act,trade_type,lots,price,comment\n"
            with open(LOG_FILE, "w") as f:
                f.write(headers)

    def _log_event(self, candle_time, pos_act, mult_act, trade_type="SIGNAL", lots=0, price=0, comment=""):
        equity = mt5.account_info().equity if mt5.initialize() else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{candle_time},{equity:.2f},{pos_act:.4f},{mult_act:.4f},{trade_type},{lots},{price},{comment}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)
        
    def _init_mt5(self):
        if not mt5.initialize():
            print(f"MT5 Init Failed: {mt5.last_error()}")
            sys.exit(1)
        
        # Detect Filling Mode (Safe way for all MT5 versions)
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            self.filling_mode = mt5.ORDER_FILLING_IOC
            return

        fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1) 
        ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        
        if symbol_info.filling_mode & fok_flag:
            self.filling_mode = mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode & ioc_flag:
            self.filling_mode = mt5.ORDER_FILLING_IOC
        else:
            self.filling_mode = mt5.ORDER_FILLING_RETURN
            
        self.peak_equity = mt5.account_info().equity
        print(f"Sniper V4 Connected. Equity: {self.peak_equity}")

    def _load_brain(self):
        def make_dummy():
            env = gym.Env()
            env.observation_space = spaces.Box(-np.inf, np.inf, (len(self.feature_columns) + 4,))
            env.action_space = spaces.Box(-1.0, 1.0, (2,))
            return env
            
        self.model = PPO.load(MODEL_PATH)
        dummy = DummyVecEnv([make_dummy])
        self.vec_normalize = VecNormalize.load(STATS_PATH, dummy)
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False
        print("Sniper Brain Loaded Successfully.")

    def _get_obs(self):
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1000)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = build_features(df, base_timeframe=BASE_TIMEFRAME_STR)
        
        last_row = df.iloc[-1]
        features = last_row[self.feature_columns].values.astype(np.float32)
        
        # Calculate PnL and Drawdown
        equity = mt5.account_info().equity
        if equity > self.peak_equity: self.peak_equity = equity
        drawdown = (equity - self.peak_equity) / (self.peak_equity + 1e-8)
        
        pnl = 0.0
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            pos = positions[0]
            entry = pos.price_open
            cur = mt5.symbol_info_tick(SYMBOL).bid if pos.type == 0 else mt5.symbol_info_tick(SYMBOL).ask
            pnl = (cur - entry) / entry * (1 if pos.type == 0 else -1)
            
        obs = np.concatenate([
            features,
            [self.last_pos],
            [self.last_mult],
            [pnl],
            [drawdown]
        ]).astype(np.float32)
        
        return self.vec_normalize.normalize_obs(obs), last_row['atr'], last_row['time']

    def execute_logic(self):
        obs, atr, candle_time = self._get_obs()
        # if candle_time == self.last_candle_time: return
        # self.last_candle_time = candle_time
        
        action, _ = self.model.predict(obs, deterministic=True)
        # Handle potential 1D or 2D action array shape
        act_flat = action.flatten()
        pos_act, mult_act = act_flat[0], act_flat[1]
        
        self.last_pos = pos_act
        self.last_mult = mult_act
        
        # print(f"[{candle_time}] Pos: {pos_act:.4f} | Mult: {mult_act:.2f} | ATR: {atr:.2f}")
        self._log_event(candle_time, pos_act, mult_act, "SIGNAL")
        
        # Sniper Logic
        net_pos = self._get_net_vol()
        
        if pos_act > NEUTRAL_THRESHOLD:
            if net_pos <= 0:
                self._close_all("Sniper Reverse", pos_act, mult_act, candle_time)
                if candle_time != self.last_candle_time:
                    print(f"[{candle_time}] Opening Buy Order: {pos_act:.4f} | Mult: {mult_act:.2f} | ATR: {atr:.2f}") 
                    self._open_order(mt5.ORDER_TYPE_BUY, atr * mult_act, pos_act, mult_act, candle_time)
                    self.last_candle_time = candle_time
        elif pos_act < -NEUTRAL_THRESHOLD:
            if net_pos >= 0:
                self._close_all("Sniper Reverse", pos_act, mult_act, candle_time)
                if candle_time != self.last_candle_time: 
                    print(f"[{candle_time}] Opening Sell Order: {pos_act:.4f} | Mult: {mult_act:.2f} | ATR: {atr:.2f}") 
                    self._open_order(mt5.ORDER_TYPE_SELL, atr * mult_act, pos_act, mult_act, candle_time)
                    self.last_candle_time = candle_time
        elif abs(pos_act) < 0.05 and net_pos != 0:
            self._close_all("Sniper TakeProfit/Neutral", pos_act, mult_act, candle_time)

    def _get_net_vol(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        vol = 0
        for p in positions:
            vol += p.volume if p.type == 0 else -p.volume
        return vol

    def _open_order(self, order_type, sl_dist, pos_act, mult_act, candle_time):
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if order_type == 0 else tick.bid
        sl = price - sl_dist if order_type == 0 else price + sl_dist
        tp = price + (sl_dist * REWARD_TO_RISK) if order_type == 0 else price - (sl_dist * REWARD_TO_RISK)
        
        if DRY_RUN:
            print(f"[DRY RUN] Would {'BUY' if order_type==0 else 'SELL'} {LOT_SIZE} @ {price} | SL: {sl:.2f} | TP: {tp:.2f}")
            self._log_event(candle_time, pos_act, mult_act, "DRY_BUY" if order_type==0 else "DRY_SELL", LOT_SIZE, price, "DryRun")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC_NUMBER,
            "comment": "Sniper V4",
            "type_filling": self.filling_mode,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"ORDER PLACED: {'BUY' if order_type==0 else 'SELL'} @ {price} | SL: {sl:.2f} | TP: {tp:.2f}")
            self._log_event(candle_time, pos_act, mult_act, "BUY" if order_type==0 else "SELL", LOT_SIZE, price, "LiveTrade")
        else:
            print(f"Order Failed: {res.comment}")
            self._log_event(candle_time, pos_act, mult_act, "ERROR", 0, 0, f"Fail: {res.comment}")

    def _close_all(self, reason, pos_act, mult_act, candle_time):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return

        if DRY_RUN:
            print(f"[DRY RUN] Would close {len(positions)} positions for {reason}")
            self._log_event(candle_time, pos_act, mult_act, "DRY_CLOSE", 0, 0, reason)
            return

        for p in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            price_close = tick.bid if p.type == 0 else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": p.volume,
                "type": 1 if p.type == 0 else 0,
                "position": p.ticket,
                "price": price_close,
                "magic": MAGIC_NUMBER,
                "comment": reason,
                "type_filling": self.filling_mode,
            }
            res = mt5.order_send(request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Closed {p.ticket} for {reason}")
                self._log_event(candle_time, pos_act, mult_act, "CLOSE", p.volume, price_close, reason)

    def run(self):
        print(f"Sniper V4 Live Monitor active for {SYMBOL}...")
        while True:
            try:
                self.execute_logic()
                time.sleep(10)
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"Runtime Error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    trader = SniperLiveV4()
    trader.run()
    mt5.shutdown()
