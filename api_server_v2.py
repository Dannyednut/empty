"""
PPO Command Center 2.1 - Centralized Trading, Training & Evaluation Hub
Features: 
- Multi-Model & Multi-TF Management
- Live MT5 Statistics & Telemetry
- Real-time Log Streaming & Evaluation Reports
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import os
import sys
import glob
import MetaTrader5 as mt5
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="PPO Command Center 2.1")

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ENV SETTINGS
# Change this to your preferred python command (e.g., "python", "py -3.10", etc.)
PYTHON_EXE = "py -3.9" 

# Process Tracker
active_tasks = {}

class TrainSpec(BaseModel):
    symbol: str = "xauusd"
    timeframe: str = "m5"
    steps: int = 500000
    model_type: str = "sniper"

class TradeSpec(BaseModel):
    symbol: str = "xauusd"
    timeframe: str = "m5"
    lots: float = 0.01
    dry_run: bool = False
    model_type: str = "sniper"

class EvalSpec(BaseModel):
    symbol: str = "xauusd"
    timeframe: str = "m5"
    model_type: str = "sniper"

class MaintenanceSpec(BaseModel):
    symbol: str = "xauusd"
    timeframe: str = "m5"
    model_type: str = "sniper"
    steps: int = 50000

# =============================================================
# ASSET & DATA DISCOVERY
# =============================================================

@app.get("/api/assets")
async def list_assets():
    assets = []
    # Discover all .zip models in hierarchical or flat models/ directory
    for f in glob.glob("models/**/*.zip", recursive=True):
        mtype = "expert" if "expert" in f.lower() else "sniper" if "sniper" in f.lower() else "unknown"
        assets.append({"name": os.path.basename(f), "type": mtype, "path": f.replace("\\", "/"), "modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()})
    
    # Discovery .onnx models for Sniper export
    for f in glob.glob("models/**/*.onnx", recursive=True):
         assets.append({"name": os.path.basename(f), "type": "onnx", "path": f.replace("\\", "/"), "modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()})
         
    return {"assets": assets}

@app.get("/api/reports")
async def list_reports():
    reports = []
    # Use forward slashes for URLs and replace backslashes from Windows glob
    for f in glob.glob("reports/*.png"):
        safe_path = f.replace("\\", "/")
        reports.append({"name": os.path.basename(f), "url": f"/api/report/view?path={safe_path}", "path": safe_path})
    return {"reports": reports}

@app.get("/api/report/view")
async def view_report(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Report not found")

# =============================================================
# LIVE STATISTICS
# =============================================================

@app.get("/api/stats")
async def get_live_stats():
    if not mt5.initialize():
        return {"error": "MT5 Not Connected"}
    
    acc = mt5.account_info()
    positions = mt5.positions_get()
    
    pos_data = []
    if positions:
        for p in positions:
            pos_data.append({
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "profit": p.profit,
                "price": p.price_open
            })
            
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "margin_free": acc.margin_free,
        "profit": acc.profit,
        "active_trades": len(pos_data),
        "positions": pos_data
    }

# =============================================================
# OPERATIONS (TRAIN, TRADE, EVAL)
# =============================================================

@app.post("/api/train/start")
async def start_training(spec: TrainSpec):
    task_id = f"train_{spec.symbol}_{spec.timeframe}_{spec.model_type}"
    if task_id in active_tasks: return {"error": "Task already running"}

    if spec.model_type == "apex_v6": script = "train_v6.py"
    elif spec.model_type == "pulse_v5": script = "train_v5.py"
    elif spec.model_type == "sniper": script = "train_v4.py"
    else: script = "train_v3.py"
    
    cmd = PYTHON_EXE.split() + ["-u", script, "--symbol", spec.symbol, "--timeframe", spec.timeframe, "--steps", str(spec.steps)]
    
    log_path = f"logs/{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    active_tasks[task_id] = {"proc": proc, "log": log_path, "type": f"Training {spec.model_type.upper()}", "start": datetime.now().isoformat()}
    return {"message": "Success", "task_id": task_id}

@app.post("/api/trade/start")
async def start_trading(spec: TradeSpec):
    task_id = f"trade_{spec.symbol}_{spec.timeframe}_{spec.model_type}"
    if task_id in active_tasks: return {"error": "Task already running"}

    if spec.model_type == "apex_v6": script = "live_sniper_v6.py"
    elif spec.model_type == "pulse_v5": script = "live_sniper_v5.py"
    elif spec.model_type == "sniper": script = "live_sniper_v4.py"
    else: script = "live_trader_mt5_v2.py"
    
    cmd = PYTHON_EXE.split() + ["-u", script, "--symbol", spec.symbol, "--tf", spec.timeframe.upper(), "--lots", str(spec.lots)]
    if spec.dry_run: cmd.extend(["--dry_run", "True"])
    
    log_path = f"logs/{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    active_tasks[task_id] = {"proc": proc, "log": log_path, "type": f"Live {spec.model_type.upper()}", "start": datetime.now().isoformat()}
    return {"message": "Success", "task_id": task_id}

@app.post("/api/trade/fleet")
async def start_fleet(dry_run: bool = False):
    task_id = "trade_fleet"
    if task_id in active_tasks: return {"error": "Fleet already sailing"}

    cmd = PYTHON_EXE.split() + ["-u", "multi_instance_trader.py", "--dry_run", str(dry_run)]
    log_path = "logs/trade_fleet.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    active_tasks[task_id] = {"proc": proc, "log": log_path, "type": "Live Multi-Fleet " + ("(DRY)" if dry_run else "(LIVE)"), "start": datetime.now().isoformat()}
    return {"message": "Trading Fleet Launched", "task_id": task_id}

@app.post("/api/eval/start")
async def start_eval(spec: EvalSpec):
    task_id = f"eval_{spec.symbol}_{spec.timeframe}_{spec.model_type}"
    if task_id in active_tasks: return {"error": "Task already running"}

    if spec.model_type == "apex_v6": script = "evaluate_v6.py"
    elif spec.model_type == "pulse_v5": script = "evaluate_v5.py"
    elif spec.model_type == "sniper": script = "evaluate_v4.py"
    else: script = "evaluate_v3.py"
    
    cmd = PYTHON_EXE.split() + ["-u", script, "--symbol", spec.symbol, "--timeframe", spec.timeframe]
    
    log_path = f"logs/{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    active_tasks[task_id] = {"proc": proc, "log": log_path, "type": f"Evaluation {spec.model_type.upper()}", "start": datetime.now().isoformat()}
    return {"message": "Evaluation Started", "task_id": task_id}

@app.post("/api/maintenance/start")
async def start_maintenance(spec: MaintenanceSpec):
    task_id = f"maint_{spec.symbol}_{spec.timeframe}_{spec.model_type}"
    if task_id in active_tasks: return {"error": "Task already running"}

    cmd = PYTHON_EXE.split() + ["-u", "retrain_pipeline.py", "--symbol", spec.symbol, "--timeframe", spec.timeframe, "--model_type", spec.model_type, "--steps", str(spec.steps)]
    
    log_path = f"logs/{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    active_tasks[task_id] = {"proc": proc, "log": log_path, "type": f"Maintenance {spec.model_type.upper()}", "start": datetime.now().isoformat()}
    return {"message": "Maintenance Duel Started", "task_id": task_id}

# =============================================================
# SYSTEM CONTROL
# =============================================================

@app.get("/api/status")
async def get_status():
    report = []
    to_delete = []
    for tid, info in active_tasks.items():
        if info["proc"].poll() is None:
            report.append({"id": tid, "type": info["type"], "start": info["start"], "status": "RUNNING"})
        else:
            to_delete.append(tid)
    for tid in to_delete: del active_tasks[tid]
    return {"active_tasks": report}

@app.post("/api/stop")
async def stop_task(task_id: str):
    if task_id == "all_traders":
        for tid in list(active_tasks.keys()):
            if "trade_" in tid: active_tasks[tid]["proc"].terminate(); del active_tasks[tid]
        return {"message": "All traders stopped"}
    if task_id in active_tasks:
        active_tasks[task_id]["proc"].terminate()
        del active_tasks[task_id]
        return {"message": "Task stopped"}
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/api/logs")
async def get_logs(task_id: str, lines: int = 50):
    log_path = None
    if task_id in active_tasks: log_path = active_tasks[task_id]["log"]
    else:
        matches = glob.glob(f"logs/*{task_id}*.log")
        if matches: log_path = matches[0]

    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return {"logs": f.readlines()[-lines:]}
    return {"logs": ["No logs available."]}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
