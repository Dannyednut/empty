"""
PPO Trading Sniper - Multi-Instance Manager
Manages multiple specialized experts (Gold, Silver, Oil) from a single bridge.
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

# Import project modules
sys.path.append('.')
from features.indicators_v2 import build_features, get_feature_columns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", type=str, default="False")
args = parser.parse_args()

ASSETS = [
    {"symbol": "XAUUSD", "model": "models/xauusd/experts/xauusd_m5_ppo_expert.zip", "stats": "models/xauusd/experts/xauusd_m5_ppo_expert_vec_normalize.pkl"},
    {"symbol": "XAGUSD", "model": "models/xagusd/experts/xagusd_m5_ppo_expert.zip", "stats": "models/xagusd/experts/xagusd_m5_ppo_expert_vec_normalize.pkl"},
]

DRY_RUN = args.dry_run.lower() == "true"
POLLING_INTERVAL = 1
MAGIC_NUMBER_BASE = 20241200
LOG_FILE = "logs/multi_asset_live_log.csv"
CONVICTION_DROP_THRESHOLD = 0.50

class ExpertInstance:
    def __init__(self, config, feature_columns):
        self.symbol = config['symbol']
        self.model_path = config['model']
        self.stats_path = config['stats']
        self.feature_columns = feature_columns
        
        self.model = None
        self.vec_normalize = None
        self.peak_conviction = 0.0
        self.last_candle_time = None
        
        # Load
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            print(f"Loaded Expert: {self.symbol}")
        else:
            print(f"Skip {self.symbol}: Model not found at {self.model_path}")

    def init_stats(self, dummy_env):
        if os.path.exists(self.stats_path):
            self.vec_normalize = VecNormalize.load(self.stats_path, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False

class MultiInstanceTrader:
    def __init__(self):
        self.feature_columns = get_feature_columns()
        self.experts = []
        self._initialize_mt5()
        
        # Setup experts
        for cfg in ASSETS:
            expert = ExpertInstance(cfg, self.feature_columns)
            if expert.model:
                expert.init_stats(DummyVecEnv([lambda: self._create_dummy_env_instance()]))
                expert.filling_mode = self._detect_filling_mode(expert.symbol)
                self.experts.append(expert)
        
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self._init_log()
                
    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            headers = "timestamp,candle_time,symbol,equity,action,peak_conv,trade_type,lots,price,comment\n"
            with open(LOG_FILE, "w") as f:
                f.write(headers)

    def _log_event(self, expert, candle_time, action, trade_type="SIGNAL", lots=0, price=0, comment=""):
        equity = mt5.account_info().equity if mt5.initialize() else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{candle_time},{expert.symbol},{equity:.2f},{action:.4f},{expert.peak_conviction:.4f},{trade_type},{lots},{price},{comment}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)

    def _initialize_mt5(self):
        if not mt5.initialize(): sys.exit(1)
        print(f"Multi-Bridge Online: {mt5.account_info().login}")

    def _detect_filling_mode(self, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: return mt5.ORDER_FILLING_IOC
        fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1) 
        ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        filling_modes = symbol_info.filling_mode
        if filling_modes & fok_flag: return mt5.ORDER_FILLING_FOK
        if filling_modes & ioc_flag: return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _create_dummy_env_instance(self):
        class SimpleEnv(gym.Env):
            def __init__(self, n_features):
                super().__init__()
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features + 3,), dtype=np.float32)
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            def reset(self, seed=None, options=None): 
                super().reset(seed=seed)
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            def step(self, action): 
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
        return SimpleEnv(len(self.feature_columns))

    def run(self):
        print(f"📡 Monitoring {len(self.experts)} assets...")
        while True:
            try:
                now = datetime.now()
                if now.minute % 5 == 0 and now.second < 10:
                    current_candle_time = now.replace(second=0, microsecond=0)
                    for expert in self.experts:
                        if current_candle_time != expert.last_candle_time:
                            self.process_expert(expert, current_candle_time)
                            expert.last_candle_time = current_candle_time
                time.sleep(POLLING_INTERVAL)
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"⚠️ Global Error: {e}")
                time.sleep(5)

    def process_expert(self, expert, candle_time):
        print(f"--- {expert.symbol} Update ---")
        rates = mt5.copy_rates_from_pos(expert.symbol, mt5.TIMEFRAME_M5, 0, 1000)
        if rates is None: return
        df = build_features(pd.DataFrame(rates).assign(time=lambda x: pd.to_datetime(x['time'], unit='s')), base_timeframe='5min')
        
        # Get Action
        last_row = df.iloc[-1]
        features = last_row[expert.feature_columns].values.astype(np.float32)
        obs = np.nan_to_num(np.concatenate([features, [0.0, 0.0, 0.0]]), nan=0.0)
        if expert.vec_normalize: obs = expert.vec_normalize.normalize_obs(obs)
            
        action, _ = expert.model.predict(obs, deterministic=True)
        target_action = action[0]
        
        self._log_event(expert, candle_time, target_action, "SIGNAL")
        if not DRY_RUN:
            self._execute_trade(expert, target_action, candle_time)

    def _execute_trade(self, expert, target_action, candle_time):
        positions = mt5.positions_get(symbol=expert.symbol)
        current_type = positions[0].type if positions else None
        
        # Manager Logic (Symbol-Level Peak Tracking & Exits)
        if current_type is not None:
            conv = abs(target_action)
            is_same_side = (target_action > 0 and current_type == 0) or (target_action < 0 and current_type == 1)
            
            if is_same_side:
                if conv > expert.peak_conviction: expert.peak_conviction = conv
                if conv < expert.peak_conviction * (1 - CONVICTION_DROP_THRESHOLD):
                    self._close_all(expert, candle_time, target_action, "PeakDrop")
                    return
            else: self._close_all(expert, candle_time, target_action, "Flip")

        if target_action > 0.2:
            if current_type != 0: self._open_order(expert, mt5.ORDER_TYPE_BUY, target_action, candle_time)
        elif target_action < -0.2:
            if current_type != 1: self._open_order(expert, mt5.ORDER_TYPE_SELL, abs(target_action), candle_time)
        elif abs(target_action) < 0.1 and current_type is not None:
            self._close_all(expert, candle_time, target_action, "Neutral")

    def _close_all(self, expert, candle_time, action, reason):
        for pos in mt5.positions_get(symbol=expert.symbol):
            tick = mt5.symbol_info_tick(expert.symbol)
            price = tick.bid if pos.type == 0 else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": expert.symbol, "volume": pos.volume,
                "type": 1 if pos.type == 0 else 0, "position": pos.ticket, "price": price,
                "deviation": 20, "magic": MAGIC_NUMBER_BASE, "comment": f"MX {reason}",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": expert.filling_mode,
            }
            res = mt5.order_send(request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"{expert.symbol} Closed: {reason}")
                self._log_event(expert, candle_time, action, "CLOSE", pos.volume, price, reason)
        expert.peak_conviction = 0.0

    def _open_order(self, expert, order_type, weight, candle_time):
        info = mt5.symbol_info(expert.symbol)
        tick = mt5.symbol_info_tick(expert.symbol)
        price = tick.ask if order_type == 0 else tick.bid
        
        # Lot sizing (Simplified for multi: 0.01 fixed or based on balance if 1 asset exists)
        lots = round(max(info.volume_min, min(1.0, 0.01)) / info.volume_step) * info.volume_step
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": expert.symbol, "volume": float(round(lots, 2)),
            "type": order_type, "price": price, "deviation": 20, "magic": MAGIC_NUMBER_BASE,
            "comment": "MultiExpert", "type_time": mt5.ORDER_TIME_GTC, "type_filling": expert.filling_mode,
        }
        dist_sl, dist_tp = price * 0.005, price * 0.015
        request["sl"] = price - dist_sl if order_type == 0 else price + dist_sl
        request["tp"] = price + dist_tp if order_type == 0 else price - dist_tp

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{expert.symbol} {'BUY' if order_type==0 else 'SELL'} @ {price}")
            self._log_event(expert, candle_time, weight, "BUY" if order_type==0 else "SELL", lots, price)
            expert.peak_conviction = abs(weight)

if __name__ == "__main__":
    trader = MultiInstanceTrader()
    trader.run()
    mt5.shutdown()
