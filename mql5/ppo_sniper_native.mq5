//+------------------------------------------------------------------+
//|                                     PPO_Sniper_Native_ONNX.mq5   |
//|                                  Copyright 2025, PPO Suite Team |
//|                                             https://localhost    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, PPO Suite Team"
#property link      "https://localhost"
#property version   "2.00"
#property strict

// Resource: Include the ONNX file directly in the EA
// Note: Ensure xauusd_ppo.onnx is in MQL5/Files/ before compiling
#resource "\\Files\\xauusd_ppo.onnx" as uchar onnx_data[]

//--- Input parameters
input double   InpNeutralThreshold = 0.2;   // Min conviction to trade
input double   InpConvictionDropThreshold = 0.5; // 50% drop from peak triggers exit
input double   InpLotSize         = 0.01;  // Fixed lot size for demo
input int      InpGMTOffset       = 0;     // GMT Offset to align with training data

#include <Trade\Trade.mqh>
CTrade trade;

//--- Internal Consts
#define FEATURE_COUNT 35
#define TOTAL_INPUTS  38 // 35 Features + 3 Env States

//--- Indicator Handles
int h_sma9, h_tema14, h_rsi14, h_macd, h_adx, h_bands, h_stoch, h_atr;
int h_rsi15m, h_ema15m, h_ema30m;

//--- Global variables
long           m_handle = INVALID_HANDLE;
float          m_last_action = 0; // Brain Memory Parity
double         m_peak_equity = 0; // Drawdown Logic
double         m_peak_conviction = 0; // Trail logic

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(_Symbol != "XAUUSD") {
      Print("❌ ERROR: Optimized for XAUUSD only.");
      return(INIT_FAILED);
   }

   // 1. Initialize ONNX
   m_handle = OnnxCreateFromBuffer(onnx_data, ONNX_DEFAULT);
   if(m_handle == INVALID_HANDLE) {
      Print("❌ ONNX Load Failed. Error: ", _LastError);
      return(INIT_FAILED);
   }

   const long input_shape[] = {1, TOTAL_INPUTS};
   const long output_shape[] = {1, 1};
   
   if(!OnnxSetInputShape(m_handle, 0, input_shape) || !OnnxSetOutputShape(m_handle, 0, output_shape)) {
      Print("❌ Shape Config Failed.");
      return(INIT_FAILED);
   }

   // 2. Initialize Handles (M5)
   h_sma9    = iMA(_Symbol, PERIOD_M5, 9, 0, MODE_SMA, PRICE_CLOSE);
   h_tema14  = iTEMA(_Symbol, PERIOD_M5, 14, 0, PRICE_CLOSE);
   h_rsi14   = iRSI(_Symbol, PERIOD_M5, 14, PRICE_CLOSE);
   h_macd    = iMACD(_Symbol, PERIOD_M5, 12, 26, 9, PRICE_CLOSE);
   h_adx     = iADX(_Symbol, PERIOD_M5, 14);
   h_bands   = iBands(_Symbol, PERIOD_M5, 20, 0, 2.0, PRICE_CLOSE);
   h_stoch   = iStochastic(_Symbol, PERIOD_M5, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   h_atr     = iATR(_Symbol, PERIOD_M5, 14);
   
   // 3. Multi-Timeframe Handles
   h_rsi15m  = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   h_ema15m  = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
   h_ema30m  = iMA(_Symbol, PERIOD_M30, 20, 0, MODE_EMA, PRICE_CLOSE);

   m_peak_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   Print("🛰️ PPO Sniper V2.0 (Handles) Initialized. System Status: GREEN.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(m_handle != INVALID_HANDLE) OnnxRelease(m_handle);
   IndicatorRelease(h_sma9);
   IndicatorRelease(h_tema14);
   IndicatorRelease(h_rsi14);
   IndicatorRelease(h_macd);
   IndicatorRelease(h_adx);
   IndicatorRelease(h_bands);
   IndicatorRelease(h_stoch);
   IndicatorRelease(h_atr);
   IndicatorRelease(h_rsi15m);
   IndicatorRelease(h_ema15m);
   IndicatorRelease(h_ema30m);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(m_handle == INVALID_HANDLE) return;
   
   // 1. Safety Guard: Ensure enough bars exist for indicators (TEMA needs ~42)
   if(iBars(_Symbol, PERIOD_M5) < 100) return;
   if(BarsCalculated(h_sma9) < 50 || BarsCalculated(h_tema14) < 50) return;

   // 2. Check for new 5M candle
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_M5, 0);
   if(current_time == last_time) return;
   last_time = current_time;

   // 3. Inference
   float inputs[TOTAL_INPUTS];
   BuildFeatureVector(inputs);

   float output[1];
   if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, inputs, output)) {
      Print("❌ Brain Error.");
      return;
   }
   
   m_last_action = output[0]; // Update memory for NEXT candle

   // 4. Execution
   ProcessAction(output[0]);
}

//+------------------------------------------------------------------+
//| Feature Engine: Handle-to-Buffer Mapping (Bar 1 for Parity)      |
//+------------------------------------------------------------------+
void BuildFeatureVector(float &inputs[])
{
   // helper shift (1 = most recent CLOSED candle)
   int s = 1; 

   // 0. diff, 1. diff_prev
   inputs[0] = (float)(GetVal(h_sma9, 0, s) - GetVal(h_tema14, 0, s));
   inputs[1] = (float)(GetVal(h_sma9, 0, s+1) - GetVal(h_tema14, 0, s+1));
   
   // 2. candle_range
   inputs[2] = (float)(iHigh(_Symbol, PERIOD_M5, s) - iLow(_Symbol, PERIOD_M5, s));
   
   // 3. RSI (4. oversold, 5. overbought)
   double rsi = GetVal(h_rsi14, 0, s);
   inputs[3] = (float)rsi;
   inputs[4] = (float)(rsi < 30 ? 1.0 : 0.0);
   inputs[5] = (float)(rsi > 70 ? 1.0 : 0.0);
   
   // 6. MACD Hist, 7. Cross
   double macd_main = GetVal(h_macd, 0, s);
   double macd_sig  = GetVal(h_macd, 1, s);
   double macd_hist = macd_main - macd_sig;
   inputs[6] = (float)macd_hist;
   inputs[7] = (float)(macd_hist > 0 ? 1.0 : -1.0);
   
   // 8. ADX
   inputs[8] = (float)GetVal(h_adx, 0, s);
   
   // 9. BB Width, 10. BB Position
   double up = GetVal(h_bands, 1, s); // 1 = Upper
   double lo = GetVal(h_bands, 2, s); // 2 = Lower
   double mid = GetVal(h_bands, 0, s); // 0 = Base
   inputs[9] = (float)((up - lo) / (mid + 1e-10));
   inputs[10] = (float)((iClose(_Symbol, PERIOD_M5, s) - lo) / (up - lo + 1e-10));
   
   // 11. Stoch_k, 12. Stoch_d
   inputs[11] = (float)GetVal(h_stoch, 0, s);
   inputs[12] = (float)GetVal(h_stoch, 1, s);
   
   // 13. ROC (10)
   double c0 = iClose(_Symbol, PERIOD_M5, s);
   double c10 = iClose(_Symbol, PERIOD_M5, s+10);
   inputs[13] = (float)((c0 - c10) / (c10 + 1e-10) * 100.0);
   
   // 14. ATR_pct, 15. Volatility
   inputs[14] = (float)(GetVal(h_atr, 0, s) / (c0 + 1e-10));
   double vol20 = CalculateVolatilityM5(20);
   inputs[15] = (float)vol20;
   
   // 16. Volume Ratio
   double v_curr = (double)iVolume(_Symbol, PERIOD_M5, s);
   double v_ma = CalculateMAVolume(20); // Calculates MA excluding current live bar
   inputs[16] = (float)(v_curr / (v_ma + 1e-10)); 
   
   // 17. Dist High, 18. Dist Low, 19. Body Ratio, 20. Doji
   double h20 = iHigh(_Symbol, PERIOD_M5, iHighest(_Symbol, PERIOD_M5, MODE_HIGH, 20, s));
   double l20 = iLow(_Symbol, PERIOD_M5, iLowest(_Symbol, PERIOD_M5, MODE_LOW, 20, s));
   inputs[17] = (float)((h20 - c0) / (c0 + 1e-10));
   inputs[18] = (float)((c0 - l20) / (c0 + 1e-10));
   double body = MathAbs(c0 - iOpen(_Symbol, PERIOD_M5, s));
   double range = iHigh(_Symbol, PERIOD_M5, s) - iLow(_Symbol, PERIOD_M5, s);
   inputs[19] = (float)(body / (range + 1e-10));
   inputs[20] = (float)(inputs[19] < 0.1 ? 1.0 : 0.0);
   
   // 21-24. Time Sin/Cos (Alignment: Monday=0 to match Python)
   MqlDateTime dt;
   datetime local_time = TimeCurrent() - (InpGMTOffset * 3600);
   TimeToStruct(local_time, dt);
   
   int py_dow = (dt.day_of_week + 6) % 7; 
   inputs[21] = (float)MathSin(2.0 * M_PI * dt.hour / 24.0);
   inputs[22] = (float)MathCos(2.0 * M_PI * dt.hour / 24.0);
   inputs[23] = (float)MathSin(2.0 * M_PI * py_dow / 7.0);
   inputs[24] = (float)MathCos(2.0 * M_PI * py_dow / 7.0);
   
   // 25-27. Sessions
   inputs[25] = (float)((dt.hour >= 0 && dt.hour < 8) ? 1.0 : 0.0);
   inputs[26] = (float)((dt.hour >= 8 && dt.hour < 16) ? 1.0 : 0.0);
   inputs[27] = (float)((dt.hour >= 16 && dt.hour < 24) ? 1.0 : 0.0);
   
   // 28-31. Regimes
   inputs[28] = (float)(inputs[8] > 25 ? 1.0 : 0.0);
   inputs[29] = (float)(inputs[8] < 20 ? 1.0 : 0.0);
   
   // Regime Vol (Simple logic for parity: if vol > last MA(vol)? No, using hard threshold as placeholder)
   inputs[30] = (float)(vol20 > 0.0005 ? 1.0 : 0.0); 
   inputs[31] = (float)(vol20 < 0.0002 ? 1.0 : 0.0);
   
   // 32. Multi-Timeframe RSI_15M, 33. Trend_15M, 34. Trend_30M
   inputs[32] = (float)GetVal(h_rsi15m, 0, 1);
   inputs[33] = (float)((iClose(_Symbol, PERIOD_M15, 1) > GetVal(h_ema15m, 0, 1)) ? 1.0 : 0.0);
   inputs[34] = (float)((iClose(_Symbol, PERIOD_M30, 1) > GetVal(h_ema30m, 0, 1)) ? 1.0 : 0.0);

   // 35. Position (MEMORY PARITY), 36. PnL (PRICE PARITY), 37. Drawdown
   inputs[35] = m_last_action; 
   
   double pnl = 0;
   if(PositionSelect(_Symbol)) {
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      if(entry > 0) {
         double cur = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         pnl = (cur - entry) / entry;
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) pnl *= -1;
      }
   }
   inputs[36] = (float)pnl;
   
   // 37. Drawdown (Peak-to-Equity logic)
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > m_peak_equity) m_peak_equity = eq;
   double dd = (m_peak_equity > 0) ? (eq - m_peak_equity) / m_peak_equity : 0;
   inputs[37] = (float)dd;

   // SCRUB: Safety check for Brain Stability
   for(int i=0; i<TOTAL_INPUTS; i++) {
      if(!MathIsValidNumber(inputs[i])) inputs[i] = 0.0f;
   }

   // DASHBOARD: Print all 38 features once per day for bit-level alignment
   static datetime last_dump = 0;
   if(MQLInfoInteger(MQL_TESTER) && (last_dump == 0 || TimeCurrent() - last_dump > 86400)) {
      last_dump = TimeCurrent();
      string dump = "� FULL CORE DUMP:\n";
      for(int i=0; i<TOTAL_INPUTS; i++) dump += StringFormat("[%d]:%.4f ", i, inputs[i]);
      Print(dump);
   }
}

//+------------------------------------------------------------------+
//| Core Performance Functions                                       |
//+------------------------------------------------------------------+
double CalculateMAVolume(int window) {
   double sum = 0;
   // Offset by 1 to skip the current unclosed bar (Bar 0)
   for(int i=1; i<=window; i++) sum += (double)iVolume(_Symbol, PERIOD_M5, i);
   return sum / window;
}

double GetVal(int handle, int buffer, int shift) {
   double val[1];
   if(CopyBuffer(handle, buffer, shift, 1, val) < 0) return 0;
   return val[0];
}

double CalculateVolatilityM5(int window) {
   double returns[];
   ArrayResize(returns, window);
   double sum = 0;
   // Start from 1 to avoid Zero-Range/Zero-Volume issues in the current bar
   for(int i=0; i<window; i++) {
      double c0 = iClose(_Symbol, PERIOD_M5, i+1);
      double c1 = iClose(_Symbol, PERIOD_M5, i+2);
      returns[i] = (c1 > 0) ? (c0 - c1) / c1 : 0;
      sum += returns[i];
   }
   double mean = sum / window;
   double sq_sum = 0;
   for(int i=0; i<window; i++) sq_sum += MathPow(returns[i] - mean, 2);
   return MathSqrt(sq_sum / (window - 1)); // Sample StdDev for Parity
}

double GetCurrentNetPos() {
   double vol = 0;
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetString(POSITION_SYMBOL)==_Symbol) {
         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) vol += PositionGetDouble(POSITION_VOLUME);
         else vol -= PositionGetDouble(POSITION_VOLUME);
      }
   }
   return vol;
}

void ProcessAction(double action) {
   double abs_conviction = MathAbs(action);
   double current_vol = GetCurrentNetPos();
   
   // CONVICTION MONITOR
   if(MQLInfoInteger(MQL_TESTER)) {
      PrintFormat("🧠 Brain Activity | Conviction: %.4f | Peak: %.4f | State: %s", action, m_peak_conviction, (current_vol != 0 ? "IN TRADE" : "FLAT"));
   }
   
   // 1. Peak Tracking & Conviction Drop Exit
   if(current_vol != 0) {
      // Check if current action direction matches current position
      bool side_matched = (action > 0 && current_vol > 0) || (action < 0 && current_vol < 0);
      
      if(side_matched) {
         if(abs_conviction > m_peak_conviction) {
            m_peak_conviction = abs_conviction;
         }
         
         // Exit if conviction drops too much from peak
         if(abs_conviction < m_peak_conviction * (1.0 - InpConvictionDropThreshold)) {
            PrintFormat("🛑 CONVICTION DROP: %.4f < %.4f peak. Closing.", abs_conviction, m_peak_conviction);
            CloseAll();
            m_peak_conviction = 0;
            return;
         }
      }
      else {
         // Direction Flip
         PrintFormat("🔄 DIRECTION FLIP: Action %.4f vs Pos %.2f. Reversing.", action, current_vol);
         CloseAll();
         m_peak_conviction = 0;
      }
   }

   double price_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Env Constants: SL 0.5%, TP 1.5%
   double sl_dist = target_price_to_points(_Symbol, 0.005);
   double tp_dist = target_price_to_points(_Symbol, 0.015);
   
   if(action > InpNeutralThreshold) {
      if(GetCurrentNetPos() <= 0) { // Check pos again as we might have closed it above
         m_peak_conviction = abs_conviction;
         double sl = price_ask - sl_dist;
         double tp = price_ask + tp_dist;
         trade.Buy(InpLotSize, _Symbol, price_ask, sl, tp, "PPO Bull");
      }
   }
   else if(action < -InpNeutralThreshold) {
      if(GetCurrentNetPos() >= 0) {
         m_peak_conviction = abs_conviction;
         double sl = price_bid + sl_dist;
         double tp = price_bid - tp_dist;
         trade.Sell(InpLotSize, _Symbol, price_bid, sl, tp, "PPO Bear");
      }
   }
   else if(abs_conviction < 0.1 && GetCurrentNetPos() != 0) {
      CloseAll();
      m_peak_conviction = 0;
   }
}

double target_price_to_points(string sym, double pct) {
   double price = SymbolInfoDouble(sym, SYMBOL_BID);
   return price * pct;
}

void CloseAll() {
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetString(POSITION_SYMBOL)==_Symbol)
         trade.PositionClose(PositionGetTicket(i));
   }
}
//+------------------------------------------------------------------+
