//+------------------------------------------------------------------+
//|                                     PPO_Sniper_Native_ONNX.mq5   |
//|                                  Copyright 2024, PPO Suite Team |
//|                                             https://localhost    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, PPO Suite Team"
#property link      "https://localhost"
#property version   "2.00"
#property strict

// Resource: Include the ONNX file directly in the EA
// Note: Ensure xauusd_ppo.onnx is in MQL5/Files/ before compiling
#resource "\\Files\\xauusd_ppo.onnx" as uchar onnx_data[]

//--- Input parameters
input double   InpNeutralThreshold = 0.2;   // Min conviction to trade
input double   InpLotSize         = 0.01;  // Fixed lot size for demo

//--- Internal Consts
#define FEATURE_COUNT 35
#define TOTAL_INPUTS  38 // 35 Features + 3 Env States

//--- Indicator Handles
int h_sma9, h_tema14, h_rsi14, h_macd, h_adx, h_bands, h_stoch, h_atr;
int h_rsi15m, h_ema15m, h_ema30m;

//--- Global variables
long           m_handle = INVALID_HANDLE;

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
   
   // Check for new 5M candle
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_M5, 0);
   if(current_time == last_time) return;
   last_time = current_time;

   float inputs[TOTAL_INPUTS];
   BuildFeatureVector(inputs);

   float output[1];
   if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, inputs, output)) {
      Print("❌ Brain Error.");
      return;
   }

   ProcessAction(output[0]);
}

//+------------------------------------------------------------------+
//| Feature Engine: Handle-to-Buffer Mapping                         |
//+------------------------------------------------------------------+
void BuildFeatureVector(float &inputs[])
{
   // Helper variables
   double buf[1], buf2[1];

   // 0. diff, 1. diff_prev
   inputs[0] = (float)(GetVal(h_sma9, 0, 0) - GetVal(h_tema14, 0, 0));
   inputs[1] = (float)(GetVal(h_sma9, 0, 1) - GetVal(h_tema14, 0, 1));
   
   // 2. candle_range
   inputs[2] = (float)(iHigh(_Symbol, PERIOD_M5, 0) - iLow(_Symbol, PERIOD_M5, 0));
   
   // 3. RSI (4. oversold, 5. overbought)
   double rsi = GetVal(h_rsi14, 0, 0);
   inputs[3] = (float)rsi;
   inputs[4] = (float)(rsi < 30 ? 1.0 : 0.0);
   inputs[5] = (float)(rsi > 70 ? 1.0 : 0.0);
   
   // 6. MACD Hist, 7. Cross
   double hist = GetVal(h_macd, SIGNAL_LINE, 0); // Histogram is signal buffer in iMACD? Logic: MAIN - SIGNAL
   double macd_main = GetVal(h_macd, MAIN_LINE, 0);
   double macd_sig = GetVal(h_macd, SIGNAL_LINE, 0);
   double macd_hist = macd_main - macd_sig;
   inputs[6] = (float)macd_hist;
   inputs[7] = (float)(macd_hist > 0 ? 1.0 : -1.0);
   
   // 8. ADX
   inputs[8] = (float)GetVal(h_adx, 0, 0);
   
   // 9. BB Width, 10. BB Position
   double up = GetVal(h_bands, UPPER_BAND, 0);
   double lo = GetVal(h_bands, LOWER_BAND, 0);
   double mid = GetVal(h_bands, BASE_LINE, 0);
   inputs[9] = (float)((up - lo) / (mid + 1e-10));
   inputs[10] = (float)((iClose(_Symbol, PERIOD_M5, 0) - lo) / (up - lo + 1e-10));
   
   // 11. Stoch_k, 12. Stoch_d
   inputs[11] = (float)GetVal(h_stoch, 0, 0);
   inputs[12] = (float)GetVal(h_stoch, 1, 0);
   
   // 13. ROC (10)
   double c0 = iClose(_Symbol, PERIOD_M5, 0);
   double c10 = iClose(_Symbol, PERIOD_M5, 10);
   inputs[13] = (float)((c0 - c10) / (c10 + 1e-10) * 100.0);
   
   // 14. ATR_pct, 15. Volatility
   inputs[14] = (float)(GetVal(h_atr, 0, 0) / c0);
   inputs[15] = (float)CalculateVolatilityM5(20);
   
   // 16. Volume Ratio
   double v_ma = CalculateMAVolume(20); 
   inputs[16] = (float)(iVolume(_Symbol, PERIOD_M5, 0) / (v_ma + 1e-10)); 
   
   // 17. Dist High, 18. Dist Low, 19. Body Ratio, 20. Doji
   double h20 = iHigh(_Symbol, PERIOD_M5, iHighest(_Symbol, PERIOD_M5, MODE_HIGH, 20, 0));
   double l20 = iLow(_Symbol, PERIOD_M5, iLowest(_Symbol, PERIOD_M5, MODE_LOW, 20, 0));
   inputs[17] = (float)((h20 - c0) / c0);
   inputs[18] = (float)((c0 - l20) / c0);
   double body = MathAbs(c0 - iOpen(_Symbol, PERIOD_M5, 0));
   double range = iHigh(_Symbol, PERIOD_M5, 0) - iLow(_Symbol, PERIOD_M5, 0);
   inputs[19] = (float)(body / (range + 1e-10));
   inputs[20] = (float)(inputs[19] < 0.1 ? 1.0 : 0.0);
   
   // 21-24. Time Sin/Cos
   MqlDateTime dt;
   TimeCurrent(dt);
   inputs[21] = (float)MathSin(2.0 * M_PI * dt.hour / 24.0);
   inputs[22] = (float)MathCos(2.0 * M_PI * dt.hour / 24.0);
   inputs[23] = (float)MathSin(2.0 * M_PI * (dt.day_of_week) / 7.0);
   inputs[24] = (float)MathCos(2.0 * M_PI * (dt.day_of_week) / 7.0);
   
   // 25-27. Sessions
   inputs[25] = (float)((dt.hour >= 0 && dt.hour < 8) ? 1.0 : 0.0);
   inputs[26] = (float)((dt.hour >= 8 && dt.hour < 16) ? 1.0 : 0.0);
   inputs[27] = (float)((dt.hour >= 16 && dt.hour < 24) ? 1.0 : 0.0);
   
   // 28-31. Regimes
   inputs[28] = (float)(inputs[8] > 25 ? 1.0 : 0.0);
   inputs[29] = (float)(inputs[8] < 20 ? 1.0 : 0.0);
   inputs[30] = (float)(inputs[15] > 0.001 ? 1.0 : 0.0); // Simple threshold Logic
   inputs[31] = (float)(inputs[15] < 0.0005 ? 1.0 : 0.0);
   
   // 32. Multi-Timeframe RSI_15M, 33. Trend_15M, 34. Trend_30M
   inputs[32] = (float)GetVal(h_rsi15m, 0, 0);
   inputs[33] = (float)(iClose(_Symbol, PERIOD_M15, 0) > GetVal(h_ema15m, 0, 0) ? 1.0 : 0.0);
   inputs[34] = (float)(iClose(_Symbol, PERIOD_M30, 0) > GetVal(h_ema30m, 0, 0) ? 1.0 : 0.0);

   // 35. Position, 36. PnL, 37. Drawdown
   inputs[35] = (float)GetCurrentNetPos();
   inputs[36] = (float)(AccountInfoDouble(ACCOUNT_EQUITY) / AccountInfoDouble(ACCOUNT_BALANCE) - 1.0);
   inputs[37] = (float)0.0;
}

//+------------------------------------------------------------------+
//| Core Performance Functions                                       |
//+------------------------------------------------------------------+
double CalculateMAVolume(int window) {
   double sum = 0;
   for(int i=0; i<window; i++) sum += (double)iVolume(_Symbol, PERIOD_M5, i);
   return sum / window;
}

double GetVal(int handle, int buffer, int shift) {
   double val[1];
   if(CopyBuffer(handle, buffer, shift, 1, val) < 0) return 0;
   return val[0];
}

double CalculateVolatilityM5(int window) {
   double sum=0, sum_sq=0;
   for(int i=0; i<window; i++) {
      double c0 = iClose(_Symbol, PERIOD_M5, i);
      double c1 = iClose(_Symbol, PERIOD_M5, i+1);
      double ret = (c1 > 0) ? (c0 - c1) / c1 : 0;
      sum += ret;
      sum_sq += ret*ret;
   }
   double mean = sum / window;
   return MathSqrt(MathAbs((sum_sq / window) - (mean*mean)));
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
   if(MathAbs(action) < InpNeutralThreshold) return;
   
   if(action > 0.5) Print("🚀 PPO BULLISH Signal: ", action);
   if(action < -0.5) Print("🩸 PPO BEARISH Signal: ", action);
}
//+------------------------------------------------------------------+
