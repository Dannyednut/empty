"""
PPO Trading Agent - Live MT5 Execution Bridge
Professional-grade bot for XAU/USD trading
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

# Parse command line arguments to override defaults
parser = argparse.ArgumentParser(description="PPO Trading Agent - Live MT5 Execution Bridge")
parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (e.g., XAUUSD)")
parser.add_argument("--tf", type=str, default="M5", choices=["M1", "M5", "M15", "M30", "H1"], help="Timeframe")
parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade (0.01-1.0)")
parser.add_argument("--lots", type=float, default=0.01, help="Lot size (fixed)")
parser.add_argument("--sl_pips", type=int, default=50, help="Stop loss in pips")
parser.add_argument("--dry_run", type=bool, default=False, help="Dry run mode")

args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = getattr(mt5, f"TIMEFRAME_{args.tf.upper()}")
BASE_TIMEFRAME_STR = '5min' if args.tf.upper() == 'M5' else '15min' if args.tf.upper() == 'M15' else '30min'
MODEL_PATH = f"models/{args.symbol.lower()}/experts/{args.symbol.lower()}_{args.tf.lower()}_ppo_expert.zip"
STATS_PATH = f"models/{args.symbol.lower()}/experts/{args.symbol.lower()}_{args.tf.lower()}_ppo_expert_vec_normalize.pkl"

# Risk Management
MAX_RISK_PER_TRADE = args.risk  # 2% of equity
LOT_SIZE_FIXED = args.lots      # Use fixed lots if risk calculation fails
MAX_SL_PIPS = args.sl_pips           # Safeguard SL

# Operational Settings
DRY_RUN = args.dry_run  # Set to False for real execution
POLLING_INTERVAL = 1  # Seconds between checks for new candle
MAGIC_NUMBER = 20241231  # Unique ID for this bot's orders
COMMENT = "PPO_Sniper_V3"
LOG_FILE = "logs/live_trading_log.csv"

# ==============================================================================
# TRADER CLASS
# ==============================================================================

class LivePPOTrader:
    def __init__(self):
        self.model = None
        self.vec_normalize = None
        self.feature_columns = get_feature_columns()
        self.last_candle_time = None
        self.filling_mode = None
        
        # Create log directory
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self._init_log()
        
        self._initialize_mt5()
        self._detect_filling_mode()
        self._load_agent()
        
    def _init_log(self):
        """Initialize CSV log with headers if not exists"""
        if not os.path.exists(LOG_FILE):
            headers = "timestamp,candle_time,equity,action,trade_type,lots,price,comment\n"
            with open(LOG_FILE, "w") as f:
                f.write(headers)

    def _log_event(self, candle_time, action, trade_type="SIGNAL", lots=0, price=0, comment=""):
        """Save event to CSV"""
        equity = mt5.account_info().equity if mt5.initialize() else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{candle_time},{equity:.2f},{action:.4f},{trade_type},{lots},{price},{comment}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)
        
    def _initialize_mt5(self):
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed, error code = {mt5.last_error()}")
            sys.exit(1)
        
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Could not get account info. Make sure you are logged in.")
            sys.exit(1)
            
        print(f"✅ Connected to MT5 Account: {account_info.login}")
        print(f"💰 Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")

    def _detect_filling_mode(self):
        """Detect supported filling mode for the symbol"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"❌ Could not get symbol info for {SYMBOL}")
            self.filling_mode = mt5.ORDER_FILLING_IOC # Fallback
            return
            
        # Get flags (fallbacks to 0 if attribute doesn't exist)
        fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1) 
        ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        
        filling_modes = symbol_info.filling_mode
        if filling_modes & fok_flag:
            self.filling_mode = mt5.ORDER_FILLING_FOK
        elif filling_modes & ioc_flag:
            self.filling_mode = mt5.ORDER_FILLING_IOC
        else:
            self.filling_mode = mt5.ORDER_FILLING_RETURN
            
        print(f"🔧 Filling mode detected: {self.filling_mode} (Flags: {filling_modes})")
        
    def _load_agent(self):
        """Load trained PPO model and normalization stats"""
        print(f"🤖 Loading PPO model: {MODEL_PATH}")
        self.model = PPO.load(MODEL_PATH)
        
        # We need a dummy env to host VecNormalize
        dummy_env = DummyVecEnv([lambda: self._create_dummy_env_instance()])
        
        if os.path.exists(STATS_PATH):
            print(f"📈 Loading normalization stats: {STATS_PATH}")
            self.vec_normalize = VecNormalize.load(STATS_PATH, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
        else:
            print("⚠️ WARNING: No normalization stats found. Accuracy might be low!")

    def _create_dummy_env_instance(self):
        """Helper to satisfy VecNormalize initialization with a proper Gymnasium Env"""
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
        """Fetch latest rates from MT5 and prepare features"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
        if rates is None:
            print(f"❌ Error fetching rates: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Build features exactly as in training
        df = build_features(df, base_timeframe=BASE_TIMEFRAME_STR)
        
        # Ensure we have the required feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        if len(available_features) != len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            print(f"⚠️ Missing columns: {missing}")
            
        return df

    def get_action(self, df):
        """Process last row of features and get model prediction"""
        last_row = df.iloc[-1]
        
        # Construct observation vector: [features] + [pos, pnl, dd]
        # For live trading, we'll fetch current state from MT5
        features = last_row[self.feature_columns].values.astype(np.float32)
        
        # Current position state
        current_pos = self._get_current_position_size() # -1 to 1 based on lots/margin
        unrealized_pnl = self._get_unrealized_pnl_pct()
        drawdown = 0.0 # Could calculate from history if needed
        
        obs = np.concatenate([features, [current_pos], [unrealized_pnl], [drawdown]])
        obs = np.nan_to_num(obs, nan=0.0)
        
        # Normalize
        if self.vec_normalize:
            obs = self.vec_normalize.normalize_obs(obs)
            
        action, _ = self.model.predict(obs, deterministic=True)
        return action[0]

    def _get_current_position_size(self):
        """Returns -1 to 1 based on current exposure in MT5"""
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return 0.0
            
        total_lots = 0
        for pos in positions:
            if pos.type == mt5.POSITION_TYPE_BUY:
                total_lots += pos.volume
            else:
                total_lots -= pos.volume
                
        # Normalize relative to an 'aggressive' max lots (e.g., 1.0 lot)
        # In training, 1.0 meant 100% equity. For live, let's keep it simple.
        # This is a simplification; a better way is to use (lots * contract_size * price) / equity
        return np.clip(total_lots / 1.0, -1.0, 1.0)

    def _get_unrealized_pnl_pct(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return 0.0
        
        equity = mt5.account_info().equity
        total_profit = sum(p.profit for p in positions)
        return total_profit / equity

    def execute_trade(self, target_action, candle_time):
        """Sync MT5 positions with target action"""
        print(f"🎯 Target Action: {target_action:.4f}")
        
        # Log the raw signal
        self._log_event(candle_time, target_action, "SIGNAL")
        
        if DRY_RUN:
            print("🕒 [DRY RUN] No orders sent to MT5.")
            return

        # 1. Close existing opposite positions if any
        self._sync_positions(target_action, candle_time)

    def _sync_positions(self, target_action, candle_time):
        """Simple execution logic"""
        positions = mt5.positions_get(symbol=SYMBOL)
        current_type = None
        if positions:
            current_type = positions[0].type 

        # Decide intent
        if target_action > 0.2: # Bullish threshold
            if current_type != mt5.POSITION_TYPE_BUY:
                self._close_all(candle_time, target_action)
                self._open_order(mt5.ORDER_TYPE_BUY, target_action, candle_time)
        elif target_action < -0.2: # Bearish threshold
            if current_type != mt5.POSITION_TYPE_SELL:
                self._close_all(candle_time, target_action)
                self._open_order(mt5.ORDER_TYPE_SELL, abs(target_action), candle_time)
        elif abs(target_action) < 0.1: # Neutral
            self._close_all(candle_time, target_action)

    def _close_all(self, candle_time, action):
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions: return
        
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
                "comment": "Close " + COMMENT,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.filling_mode,
            }
            res = mt5.order_send(request)
            if res is None:
                err = mt5.last_error()
                print(f"❌ Close [None] failure: {err}")
                self._log_event(candle_time, action, "CLOSE_FAILED_NONE", pos.volume, price_close, f"Err: {err}")
            elif res.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"⚠️ Close failed: {res.comment}")
            else:
                print(f"✅ Closed position {pos.ticket}")
                self._log_event(candle_time, action, "CLOSE", pos.volume, price_close, f"Pos: {pos.ticket}")

    def _open_order(self, order_type, weight, candle_time):
        """Open a new position with user-specific rule: 0.01 lot per $1000 balance, max 1.0"""
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"❌ Could not get symbol info for {SYMBOL}")
            return

        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # 1. User rule: 0.01 lot per $1000 of account equity, scaled by conviction
        equity = mt5.account_info().equity
        raw_lots = (equity / 1000.0) * 0.01 * weight
        
        # 2. Hard cap at 1.0 lot as per user request
        raw_lots = min(raw_lots, 1.0)
        
        # 3. Respect broker constraints (Step/Min/Max)
        lot_min = symbol_info.volume_min
        lot_max = min(symbol_info.volume_max, 1.0) # Respect user's 1.0 max even if broker allows more
        lot_step = symbol_info.volume_step
        
        # Round to the nearest step
        lots = round(raw_lots / lot_step) * lot_step
        
        # Final safety clip
        lots = max(lot_min, min(lot_max, lots))
        lots = round(lots, 2) # Safety round
        
        print(f"📦 Gold Lot Rule Applied: {lots:.2f} (Balance: ${equity:.2f}, Weight: {weight:.2f})")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(lots), # Ensure it's a float
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode,
        }
        
        sl_dist = price * 0.005
        tp_dist = price * 0.015
        if order_type == mt5.ORDER_TYPE_BUY:
            request["sl"] = price - sl_dist
            request["tp"] = price + tp_dist
        else:
            request["sl"] = price + sl_dist
            request["tp"] = price - tp_dist

        res = mt5.order_send(request)
        if res is None:
            err = mt5.last_error()
            print(f"❌ Order [None] failure: {err}")
            self._log_event(candle_time, weight, "ORDER_FAILED_NONE", lots, price, f"Err: {err}")
        elif res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {res.comment}")
            self._log_event(candle_time, weight, "ORDER_FAILED", lots, price, res.comment)
        else:
            t_str = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"🚀 {SYMBOL} {t_str} {lots} lots @ {price}")
            self._log_event(candle_time, weight, t_str, lots, price, f"SL: {request['sl']:.2f} TP: {request['tp']:.2f}")

    def run(self):
        """Main loop - synchronized with candle closes"""
        print(f"\n🚀 Bot started in {'DRY RUN' if DRY_RUN else 'LIVE'} mode.")
        print(f"📡 Tracking {SYMBOL} M5 candles...")
        
        while True:
            try:
                # 1. Sync check
                now = datetime.now()
                # If we are in the first 10 seconds of a new 5-minute bar
                if now.minute % 5 == 0 and now.second < 10:
                    current_candle_time = now.replace(second=0, microsecond=0)
                    
                    if current_candle_time != self.last_candle_time:
                        print(f"\n🔔 New candle detected: {current_candle_time}")
                        
                        # Wait a moment for broker to update bar data
                        time.sleep(2)
                        
                        # Fetch and predict
                        df = self.fetch_live_data()
                        if df is not None:
                            action = self.get_action(df)
                            self.execute_trade(action, current_candle_time)
                            self.last_candle_time = current_candle_time
                        
                time.sleep(POLLING_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    trader = LivePPOTrader()
    trader.run()
    mt5.shutdown()
