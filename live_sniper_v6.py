"""
PPO Sniper V6 - Apex Bridge
Ultra-Sensitive Neural Execution Engine.
Focuses on earlier exits and reversal anticipation.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import time
import os
import sys
from datetime import datetime
import argparse

# Project imports
sys.path.append('.')
from features.indicators_v2 import build_features, get_feature_columns
from env.trading_env_v6_apex import SniperApexEnvV6

# ==============================================================================
# CONFIGURATION
# ==============================================================================
parser = argparse.ArgumentParser(description="PPO Sniper V6 Apex - Live execution")
parser.add_argument("--symbol", type=str, default="XAUUSD")
parser.add_argument("--lots", type=float, default=0.01)
parser.add_argument("--dry_run", type=str, default="False")
parser.add_argument("--tf", type=str, default="M1")
args = parser.parse_args()

SYMBOL = args.symbol
DRY_RUN = args.dry_run.lower() == "true"
LOTS = args.lots
TIMEFRAME = mt5.TIMEFRAME_M1
POLLING_INTERVAL = 3 

MODEL_PATH = f"models/{SYMBOL.lower()}/sniper_v6/{SYMBOL.lower()}_m1_apex_v6_expert.zip"
STATS_PATH = f"models/{SYMBOL.lower()}/sniper_v6/{SYMBOL.lower()}_m1_apex_v6_vec_normalize.pkl"
MAGIC_NUMBER = 20261111
LOG_FILE = f"logs/sniper_v6_live_{SYMBOL.lower()}.csv"

class SniperApexV6:
    def __init__(self):
        self.model = None
        self.vec_normalize = None
        self.last_action = np.zeros(3) # [Pos, Mult, Flush]
        self.peak_equity = 0.0
        self.stagnation = 0.0
        self.peak_pnl = 0.0
        self.filling_mode = None
        
        self._init_mt5()
        self._load_brain()
        self._init_log()

    def _init_mt5(self):
        if not mt5.initialize():
            print(f"MT5 Init Failed: {mt5.last_error()}")
            sys.exit(1)
        
        info = mt5.symbol_info(SYMBOL)
        if not info:
            print(f"Symbol {SYMBOL} not found.")
            sys.exit(1)
            
        fok = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
        ioc = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        if info.filling_mode & fok: self.filling_mode = mt5.ORDER_FILLING_FOK
        elif info.filling_mode & ioc: self.filling_mode = mt5.ORDER_FILLING_IOC
        else: self.filling_mode = mt5.ORDER_FILLING_RETURN
        
        self.peak_equity = mt5.account_info().equity
        print(f"Apex V6 Survivalist Active. Equity: ${self.peak_equity:,.2f}")

    def _load_brain(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Brain not found at {MODEL_PATH}")
            sys.exit(1)
            
        def make_dummy():
            f_cols = get_feature_columns()
            dummy_data = pd.DataFrame(np.zeros((3000, len(f_cols) + 5)), columns=f_cols + ['open','high','low','close','tick_volume'])
            return SniperApexEnvV6(dummy_data, f_cols)
            
        self.model = PPO.load(MODEL_PATH)
        dummy = DummyVecEnv([make_dummy])
        self.vec_normalize = VecNormalize.load(STATS_PATH, dummy)
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False
        print("Apex Survival Brain Synchronized.")

    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                f.write("timestamp,price,pos_act,flush_act,pnl,stagnation,dd\n")

    def _get_obs(self):
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 800)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        df_feat = build_features(df, base_timeframe='1min')
        last_row = df_feat.iloc[-1]
        
        equity = mt5.account_info().equity
        if equity > self.peak_equity: self.peak_equity = equity
        dd = (equity - self.peak_equity) / (self.peak_equity + 1e-9)
        
        pnl = 0.0
        pos_mt5 = mt5.positions_get(symbol=SYMBOL)
        if pos_mt5:
            p = pos_mt5[0]
            cur = mt5.symbol_info_tick(SYMBOL).bid if p.type == 0 else mt5.symbol_info_tick(SYMBOL).ask
            pnl = (cur - p.price_open) / p.price_open * (1 if p.type == 0 else -1)
            if pnl > self.peak_pnl:
                self.peak_pnl = pnl
                self.stagnation = 0
            else:
                self.stagnation += 0.5 # Fractional increase per poll
        else:
            self.peak_pnl = 0.0
            self.stagnation = 0.0

        now = datetime.now()
        pulse_phase = (now.second // 20) / 2.0
        
        obs = np.concatenate([
            last_row[get_feature_columns()].values.astype(np.float32),
            [self.last_action[0]],
            [self.last_action[1]],
            [pnl],
            [dd],
            [self.stagnation / 50.0],
            [pulse_phase]
        ]).astype(np.float32)
        
        return self.vec_normalize.normalize_obs(obs), last_row['atr'], pnl, dd

    def execute_logic(self):
        obs, atr, pnl, dd = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        act_flat = action.flatten()
        pos_act, mult_act, flush_act = act_flat[0], act_flat[1], act_flat[2]
        
        self.last_action = act_flat
        
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.bid
        
        positions = mt5.positions_get(symbol=SYMBOL)
        net_pos = 0 
        if positions: net_pos = 1 if positions[0].type == 0 else -1
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] P:{pos_act:+.2f} | F:{flush_act:.2f} | PnL:{pnl:+.3%} | Stag:{self.stagnation:.1f}")

        # 1. Apex Flush Logic (Emergency Survival)
        if flush_act > 0.8 and net_pos != 0:
            print(f"APEX FLUSH TRIGGERED! Brain panic detected. Closing positions.")
            self._close_all("Apex_Flush")
            return

        # 2. Reversal / Entry Logic
        if pos_act > 0.15: # Long Signal
            if net_pos <= 0:
                self._close_all("Apex_Reverse")
                self._open_order(mt5.ORDER_TYPE_BUY, price, atr * mult_act)
        elif pos_act < -0.15: # Short Signal
            if net_pos >= 0:
                self._close_all("Apex_Reverse")
                self._open_order(mt5.ORDER_TYPE_SELL, price, atr * mult_act)
        elif abs(pos_act) < 0.05 and net_pos != 0: # Neural Neutral
            self._close_all("Apex_Neural_Exit")

    def _open_order(self, otype, price, dist):
        if DRY_RUN:
            print(f"[DRY RUN] Would {'BUY' if otype==0 else 'SELL'} at {price}")
            return
            
        sl = price - dist if otype == 0 else price + dist
        tp = price + (dist * 2.8) if otype == 0 else price - (dist * 2.8)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOTS,
            "type": otype,
            "price": price,
            "sl": float(round(sl, 2)),
            "tp": float(round(tp, 2)),
            "magic": MAGIC_NUMBER,
            "comment": "ApexV6_Survival",
            "type_filling": self.filling_mode,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Order Executed: {SYMBOL} {'BUY' if otype==0 else 'SELL'} at {price}")
        else:
            print(f"Order Failed: {res.comment}")

    def _close_all(self, reason):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return
        if DRY_RUN:
            print(f"[DRY RUN] Closing for {reason}")
            return
            
        for p in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            pclose = tick.bid if p.type == 0 else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": p.volume,
                "type": 1 if p.type == 0 else 0,
                "position": p.ticket,
                "price": pclose,
                "magic": MAGIC_NUMBER,
                "comment": reason,
                "type_filling": self.filling_mode,
            }
            mt5.order_send(req)
        print(f"Positions Closed: {reason}")
        self.stagnation = 0

    def run(self):
        while True:
            try:
                self.execute_logic()
                time.sleep(POLLING_INTERVAL)
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"Apex Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = SniperApexV6()
    bot.run()
    mt5.shutdown()
