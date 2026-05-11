//+------------------------------------------------------------------+
//|                                     PPO_Sniper_V4_Adaptive.mq5    |
//|                                  Copyright 2026, PPO Suite Team  |
//|                                             https://localhost     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, PPO Suite Team"
#property link      "https://localhost"
#property version   "4.00"
#property strict

// Resource: Include the V4 ONNX file (ensure it exists in MQL5/Files)
#resource "\\Files\\xauusd_sniper_v4.onnx" as uchar onnx_data[]

//--- Input parameters
input double   InpNeutralThreshold = 0.15;  // More aggressive threshold for v4
input double   InpLotSize         = 0.01;   // Fixed lot size
input int      InpGMTOffset       = 0;      // GMT Offset

#include <Trade\Trade.mqh>
CTrade trade;

//--- Internal Consts
#define FEATURE_COUNT 35
#define TOTAL_INPUTS  39 // v4 adds Multiplier to memory (35 + PrevPos + PrevMult + PnL + DD)

//--- Indicator Handles
int h_sma9, h_tema14, h_rsi14, h_macd, h_adx, h_bands, h_stoch, h_atr;
int h_rsi15m, h_ema15m, h_ema30m;

//--- Global variables
long           m_handle = INVALID_HANDLE;
float          m_last_pos = 0;   // Brain Memory: Direction
float          m_last_mult = 1.0; // Brain Memory: Volatility Scale
double         m_peak_equity = 0; 

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(_Symbol != "XAUUSD") {
      Print("❌ ERROR: Optimized for XAUUSD only.");
      return(INIT_FAILED);
   }

   m_handle = OnnxCreateFromBuffer(onnx_data, ONNX_DEFAULT);
   if(m_handle == INVALID_HANDLE) {
      Print("❌ ONNX Load Failed. Error: ", _LastError);
      return(INIT_FAILED);
   }

   const long input_shape[] = {1, TOTAL_INPUTS}; // 39 
   const long output_shape[] = {1, 2};            // [Pos, Mult]
   
   if(!OnnxSetInputShape(m_handle, 0, input_shape) || !OnnxSetOutputShape(m_handle, 0, output_shape)) {
      Print("❌ V4 Shape Config Failed.");
      return(INIT_FAILED);
   }

   // Initialize Handles
   h_sma9    = iMA(_Symbol, PERIOD_M5, 9, 0, MODE_SMA, PRICE_CLOSE);
   h_tema14  = iTEMA(_Symbol, PERIOD_M5, 14, 0, PRICE_CLOSE);
   h_rsi14   = iRSI(_Symbol, PERIOD_M5, 14, PRICE_CLOSE);
   h_macd    = iMACD(_Symbol, PERIOD_M5, 12, 26, 9, PRICE_CLOSE);
   h_adx     = iADX(_Symbol, PERIOD_M5, 14);
   h_bands   = iBands(_Symbol, PERIOD_M5, 20, 0, 2.0, PRICE_CLOSE);
   h_stoch   = iStochastic(_Symbol, PERIOD_M5, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   h_atr     = iATR(_Symbol, PERIOD_M5, 14);
   h_rsi15m  = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   h_ema15m  = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
   h_ema30m  = iMA(_Symbol, PERIOD_M30, 20, 0, MODE_EMA, PRICE_CLOSE);

   m_peak_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   Print("🏹 PPO SNIPER V4 (ADAPTIVE) ONLINE. Deployment Mode: PURE ALPHA.");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(m_handle != INVALID_HANDLE) OnnxRelease(m_handle);
}

void OnTick()
{
   if(m_handle == INVALID_HANDLE) return;
   if(iBars(_Symbol, PERIOD_M5) < 100) return;

   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_M5, 0);
   if(current_time == last_time) return;
   last_time = current_time;

   // 1. Inference
   float inputs[TOTAL_INPUTS];
   BuildFeatureVector(inputs);

   float output[2]; 
   if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, inputs, output)) return;
   
   m_last_pos = output[0];
   m_last_mult = output[1];

   // 2. Execution
   ProcessActionV4(output[0], output[1]);
}

void BuildFeatureVector(float &inputs[])
{
   int s = 1; 
   inputs[0] = (float)(GetVal(h_sma9,0,s)-GetVal(h_tema14,0,s));
   inputs[1] = (float)(GetVal(h_sma9,0,s+1)-GetVal(h_tema14,0,s+1));
   inputs[2] = (float)(iHigh(_Symbol,PERIOD_M5,s)-iLow(_Symbol,PERIOD_M5,s));
   double rsi = GetVal(h_rsi14,0,s);
   inputs[3] = (float)rsi;
   inputs[4] = (float)(rsi<30?1:0);
   inputs[5] = (float)(rsi>70?1:0);
   double mh = GetVal(h_macd,0,s)-GetVal(h_macd,1,s);
   inputs[6] = (float)mh;
   inputs[7] = (float)(mh>0?1:-1);
   inputs[8] = (float)GetVal(h_adx,0,s);
   double up=GetVal(h_bands,1,s), lo=GetVal(h_bands,2,s), mid=GetVal(h_bands,0,s);
   inputs[9] = (float)((up-lo)/(mid+1e-10));
   inputs[10] = (float)((iClose(_Symbol,PERIOD_M5,s)-lo)/(up-lo+1e-10));
   inputs[11] = (float)GetVal(h_stoch,0,s);
   inputs[12] = (float)GetVal(h_stoch,1,s);
   double c0=iClose(_Symbol,PERIOD_M5,s), c10=iClose(_Symbol,PERIOD_M5,s+10);
   inputs[13] = (float)((c0-c10)/(c10+1e-10)*100.0);
   inputs[14] = (float)(GetVal(h_atr,0,s)/(c0+1e-10));
   inputs[15] = (float)CalculateVolatilityM5(20);
   inputs[16] = (float)((double)iVolume(_Symbol,PERIOD_M5,s)/(CalculateMAVolume(20)+1e-10));
   double h20=iHigh(_Symbol,PERIOD_M5,iHighest(_Symbol,PERIOD_M5,MODE_HIGH,20,s));
   double l20=iLow(_Symbol,PERIOD_M5,iLowest(_Symbol,PERIOD_M5,MODE_LOW,20,s));
   inputs[17] = (float)((h20-c0)/(c0+1e-10));
   inputs[18] = (float)((c0-l20)/(c0+1e-10));
   inputs[19] = (float)((MathAbs(c0-iOpen(_Symbol,PERIOD_M5,s)))/(inputs[2]+1e-10));
   inputs[20] = (float)(inputs[19]<0.1?1:0);

   MqlDateTime dt; TimeToStruct(TimeCurrent()-(InpGMTOffset*3600), dt);
   int py_dow = (dt.day_of_week+6)%7;
   inputs[21] = (float)MathSin(2.0*M_PI*dt.hour/24.0);
   inputs[22] = (float)MathCos(2.0*M_PI*dt.hour/24.0);
   inputs[23] = (float)MathSin(2.0*M_PI*py_dow/7.0);
   inputs[24] = (float)MathCos(2.0*M_PI*py_dow/7.0);
   inputs[25] = (float)((dt.hour<8)?1:0);
   inputs[26] = (float)((dt.hour>=8&&dt.hour<16)?1:0);
   inputs[27] = (float)((dt.hour>=16)?1:0);
   inputs[28] = (float)(inputs[8]>25?1:0);
   inputs[29] = (float)(inputs[8]<20?1:0);
   inputs[30] = (float)(inputs[15]>0.0005?1:0);
   inputs[31] = (float)(inputs[15]<0.0002?1:0);
   inputs[32] = (float)GetVal(h_rsi15m,0,1);
   inputs[33] = (float)((iClose(_Symbol,PERIOD_M15,1)>GetVal(h_ema15m,0,1))?1:0);
   inputs[34] = (float)((iClose(_Symbol,PERIOD_M30,1)>GetVal(h_ema30m,0,1))?1:0);

   // State Memory (v4 Indexing)
   inputs[35] = m_last_pos;
   inputs[36] = m_last_mult;
   
   double pnl = 0;
   if(PositionSelect(_Symbol)) {
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
      pnl = (cur-entry)/entry;
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL) pnl *= -1;
   }
   inputs[37] = (float)pnl;
   
   double eq=AccountInfoDouble(ACCOUNT_EQUITY); if(eq>m_peak_equity) m_peak_equity=eq;
   inputs[38] = (float)((m_peak_equity>0)?(eq-m_peak_equity)/m_peak_equity:0);

   for(int i=0; i<TOTAL_INPUTS; i++) if(!MathIsValidNumber(inputs[i])) inputs[i]=0;
}

void ProcessActionV4(double pos_action, double atr_mult) {
   double net_pos = GetCurrentNetPos();
   double price_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = GetVal(h_atr, 0, 1);
   
   // DYNAMIC VOLATILITY STOPS
   double sl_dist = atr * atr_mult;
   double tp_dist = sl_dist * 2.5; // Matches v4 Env RR of 1:2.5

   if(pos_action > InpNeutralThreshold) {
      if(net_pos <= 0) {
         if(net_pos < 0) CloseAll();
         trade.Buy(InpLotSize, _Symbol, price_ask, price_ask-sl_dist, price_ask+tp_dist, "Sniper Bull");
      }
   }
   else if(pos_action < -InpNeutralThreshold) {
      if(net_pos >= 0) {
         if(net_pos > 0) CloseAll();
         trade.Sell(InpLotSize, _Symbol, price_bid, price_bid+sl_dist, price_bid-tp_dist, "Sniper Bear");
      }
   }
   else if(MathAbs(pos_action) < 0.05 && net_pos != 0) {
      CloseAll();
   }
}

// Helpers
double GetVal(int h, int b, int s) { double v[1]; return (CopyBuffer(h,b,s,1,v)>0)?v[0]:0; }
double CalculateMAVolume(int w) { double s=0; for(int i=1;i<=w;i++) s+=(double)iVolume(_Symbol,PERIOD_M5,i); return s/w; }
double CalculateVolatilityM5(int w) {
   double s=0, rets[]; ArrayResize(rets,w);
   for(int i=0;i<w;i++){ double c0=iClose(_Symbol,PERIOD_M5,i+1), c1=iClose(_Symbol,PERIOD_M5,i+2); rets[i]=(c1>0)?(c0-c1)/c1:0; s+=rets[i]; }
   double m=s/w, sq=0; for(int i=0;i<w;i++) sq+=MathPow(rets[i]-m,2); return MathSqrt(sq/(w-1));
}
double GetCurrentNetPos() {
   double v=0; for(int i=PositionsTotal()-1;i>=0;i--) {
      if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetString(POSITION_SYMBOL)==_Symbol)
         v += (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)?PositionGetDouble(POSITION_VOLUME):-PositionGetDouble(POSITION_VOLUME);
   } return v;
}
void CloseAll() { for(int i=PositionsTotal()-1;i>=0;i--) if(PositionSelectByTicket(PositionGetTicket(i)) && PositionGetString(POSITION_SYMBOL)==_Symbol) trade.PositionClose(PositionGetTicket(i)); }
