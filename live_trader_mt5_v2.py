"""
PPO Trading Agent - Live MT5 Execution Bridge V2
Featuring "Conviction Drop" Exit Logic for Gold Trading
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
from datetime import datetime, timedelta

# Import project modules
sys.path.append('.')
from features.indicators_v2 import build_features, get_feature_columns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
import argparse
parser = argparse.ArgumentParser(description="PPO Trading Agent - Live MT5 Execution Bridge")
parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (e.g., XAUUSD)")
parser.add_argument("--tf", type=str, default="M5", choices=["M1", "M5", "M15", "M30", "H1"], help="Timeframe")
parser.add_argument("--lots", type=float, default=0.01, help="Lot size (fixed)")
parser.add_argument("--dry_run", type=bool, default=False, help="Dry run mode")
args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = getattr(mt5, f"TIMEFRAME_{args.tf.upper()}")
BASE_TIMEFRAME_STR = '5min' if args.tf.upper() == 'M5' else '15min' if args.tf.upper() == 'M15' else '30min'
MODEL_PATH = f"models/{args.symbol.lower()}/experts/{args.symbol.lower()}_{args.tf.lower()}_ppo_expert.zip"
STATS_PATH = f"models/{args.symbol.lower()}/experts/{args.symbol.lower()}_{args.tf.lower()}_ppo_expert_vec_normalize.pkl"

# Operational Settings
DRY_RUN = args.dry_run  # Set to False for real execution
POLLING_INTERVAL = 1  # Seconds between checks for new candle
MAGIC_NUMBER = 20241231  # Unique ID for this bot's orders
COMMENT = "PPO_Sniper_V2"
LOG_FILE = "logs/live_trading_v2_log.csv"

# Advanced Exit Logic
CONVICTION_DROP_THRESHOLD = 0.50  # 50% drop from peak triggers exit
MIN_CONVICTION_THRESHOLD = 0.20   # Entry threshold
NEUTRAL_THRESHOLD = 0.10          # Hard neutral threshold

# ==============================================================================
# TRADER CLASS
# ==============================================================================

class LivePPOTraderV2:
    def __init__(self):
        self.model = None
        self.vec_normalize = None
        self.feature_columns = get_feature_columns()
        self.last_candle_time = None
        self.filling_mode = None
        
        # State tracking for V2 Exit Logic
        self.peak_conviction = 0.0
        self.current_pos_ticket = None
        
        # Create log directory
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self._init_log()
        
        self._initialize_mt5()
        self._detect_filling_mode()
        self._load_agent()
        
    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            headers = "timestamp,candle_time,equity,action,peak_conv,trade_type,lots,price,comment\n"
            with open(LOG_FILE, "w") as f:
                f.write(headers)

    def _log_event(self, candle_time, action, trade_type="SIGNAL", lots=0, price=0, comment=""):
        equity = mt5.account_info().equity if mt5.initialize() else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{candle_time},{equity:.2f},{action:.4f},{self.peak_conviction:.4f},{trade_type},{lots},{price},{comment}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)
        
    def _initialize_mt5(self):
        if not mt5.initialize():
            print(f"MT5 initialization failed, error code = {mt5.last_error()}")
            sys.exit(1)
        account_info = mt5.account_info()
        if account_info is None:
            sys.exit(1)
        print(f"Connected to MT5: {account_info.login}")
        print(f"Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")

    def _detect_filling_mode(self):
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            self.filling_mode = mt5.ORDER_FILLING_IOC
            return
        fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1) 
        ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        filling_modes = symbol_info.filling_mode
        if filling_modes & fok_flag:
            self.filling_mode = mt5.ORDER_FILLING_FOK
        elif filling_modes & ioc_flag:
            self.filling_mode = mt5.ORDER_FILLING_IOC
        else:
            self.filling_mode = mt5.ORDER_FILLING_RETURN
        print(f"Filling mode: {self.filling_mode}")
        
    def _load_agent(self):
        self.model = PPO.load(MODEL_PATH)
        dummy_env = DummyVecEnv([lambda: self._create_dummy_env_instance()])
        if os.path.exists(STATS_PATH):
            self.vec_normalize = VecNormalize.load(STATS_PATH, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
        print("Agent Loaded Ready.")

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

    def fetch_live_data(self, n_bars=1000):
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
        if rates is None: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return build_features(df, base_timeframe=BASE_TIMEFRAME_STR)

    def get_action(self, df):
        last_row = df.iloc[-1]
        features = last_row[self.feature_columns].values.astype(np.float32)
        current_pos = self._get_current_position_size()
        unrealized_pnl = self._get_unrealized_pnl_pct()
        obs = np.concatenate([features, [current_pos], [unrealized_pnl], [0.0]])
        obs = np.nan_to_num(obs, nan=0.0)
        if self.vec_normalize:
            obs = self.vec_normalize.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return action[0]

    def _get_current_position_size(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return 0.0
        total_lots = 0
        for pos in positions:
            if pos.type == mt5.POSITION_TYPE_BUY: total_lots += pos.volume
            else: total_lots -= pos.volume
        return np.clip(total_lots / 1.0, -1.0, 1.0)

    def _get_unrealized_pnl_pct(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return 0.0
        equity = mt5.account_info().equity
        total_profit = sum(p.profit for p in positions)
        return total_profit / equity

    def execute_trade(self, target_action, candle_time):
        print(f"Target Action: {target_action:.4f} | Peak: {self.peak_conviction:.4f}")
        self._log_event(candle_time, target_action, "SIGNAL")
        
        if DRY_RUN:
            print("DRY RUN - No trade executed")
            return

        self._sync_positions_v2(target_action, candle_time)

    def _sync_positions_v2(self, target_action, candle_time):
        """V2 Logic: Detects conviction drop and manages peak tracking"""
        positions = mt5.positions_get(symbol=SYMBOL)
        current_type = positions[0].type if positions else None
        
        # 1. Update Peak Conviction if we are in a trade
        if current_type is not None:
            conviction = abs(target_action)
            # If model is still pointing in the SAME direction
            side_matched = (target_action > 0 and current_type == mt5.POSITION_TYPE_BUY) or \
                           (target_action < 0 and current_type == mt5.POSITION_TYPE_SELL)
            
            if side_matched:
                if conviction > self.peak_conviction:
                    print(f"New Peak Conviction reached: {conviction:.4f} (Prev: {self.peak_conviction:.4f})")
                    self.peak_conviction = conviction
                
                # CHECK FOR CONVICTION DROP EXIT
                if conviction < self.peak_conviction * (1 - CONVICTION_DROP_THRESHOLD):
                    print(f"CONVICTION DROP DETECTED! Current {conviction:.4f} < {self.peak_conviction:.4f} drop limit. Exiting.")
                    self._close_all(candle_time, target_action, f"PeakDrop: {self.peak_conviction:.2f}->{conviction:.2f}")
                    return # Exit triggered, don't re-open same bar
            else:
                # Model changed direction completely (e.g., target say Sell while we are Long)
                print(f"Direction Change: Target {target_action:.4f} vs Pos {current_type}. Flipping.")
                self._close_all(candle_time, target_action, "DirFlip")

        # 2. Decide Entry or Flipping
        if target_action > MIN_CONVICTION_THRESHOLD:
            if current_type != mt5.POSITION_TYPE_BUY:
                self._open_order(mt5.ORDER_TYPE_BUY, target_action, candle_time)
        elif target_action < -MIN_CONVICTION_THRESHOLD:
            if current_type != mt5.POSITION_TYPE_SELL:
                self._open_order(mt5.ORDER_TYPE_SELL, abs(target_action), candle_time)
        elif abs(target_action) < NEUTRAL_THRESHOLD:
            if current_type is not None:
                self._close_all(candle_time, target_action, "NeutralExit")

    def _close_all(self, candle_time, action, reason=""):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return

        if DRY_RUN:
            print(f"[DRY RUN] Would close {len(positions)} positions for {reason}")
            self._log_event(candle_time, action, "DRY_CLOSE", 0, 0, reason)
            return

        for pos in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price_close = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": price_close,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": f"V2 {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.filling_mode,
            }
            res = mt5.order_send(request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Closed {pos.ticket} for {reason}")
                self._log_event(candle_time, action, "CLOSE", pos.volume, price_close, reason)
        
        # Reset Peak state after closing
        self.peak_conviction = 0.0

    def _open_order(self, order_type, weight, candle_time):
        symbol_info = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Determine lots: prioritize --lots argument if > 0, else use dynamic equity-based scaling
        if args.lots > 0:
            lots = args.lots
        else:
            equity = mt5.account_info().equity
            raw_lots = (equity / 1000.0) * 0.01 * min(weight, 1.0)
            lots = round(max(symbol_info.volume_min, min(1.0, raw_lots)) / symbol_info.volume_step) * symbol_info.volume_step
        
        if DRY_RUN:
            print(f"[DRY RUN] Would {'BUY' if order_type==0 else 'SELL'} {lots:.2f} {SYMBOL} (Conviction: {weight:.2f})")
            self._log_event(candle_time, weight, "DRY_OPEN", lots, price, "DryRun")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(round(lots, 2)),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode,
        }
        
        sl_dist, tp_dist = price * 0.005, price * 0.015
        if order_type == mt5.ORDER_TYPE_BUY:
            request["sl"], request["tp"] = price - sl_dist, price + tp_dist
        else:
            request["sl"], request["tp"] = price + sl_dist, price - tp_dist

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{SYMBOL} {'BUY' if order_type==0 else 'SELL'} {lots:.2f} @ {price}")
            self._log_event(candle_time, weight, "BUY" if order_type==0 else "SELL", lots, price, "OpenV2")
            # Initialize peak conviction for the new trade
            self.peak_conviction = abs(weight)

    def run(self):
        print(f"\nBot V2 started. Monitoring {SYMBOL} M5...")
        while True:
            try:
                now = datetime.now()
                if now.minute % 5 == 0 and now.second < 10:
                    current_candle_time = now.replace(second=0, microsecond=0)
                    if current_candle_time != self.last_candle_time:
                        time.sleep(2)
                        df = self.fetch_live_data()
                        if df is not None:
                            action = self.get_action(df)
                            self.execute_trade(action, current_candle_time)
                            self.last_candle_time = current_candle_time
                time.sleep(POLLING_INTERVAL)
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    trader = LivePPOTraderV2()
    trader.run()
    mt5.shutdown()
