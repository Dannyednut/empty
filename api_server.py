"""
PPO Trading Sniper - Command Center API
Centralized orchestration for the entire trading suite.
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import subprocess
import os
import sys
import psutil
import MetaTrader5 as mt5
from datetime import datetime
import json

app = FastAPI(title="PPO Sniper Command Center")

# =============================================================
# PROCESS MANAGEMENT
# =============================================================
# Global state to track background processes
processes = {
    "trader": None,
    "trainer": None,
    "pipeline": None
}

class TrainRequest(BaseModel):
    symbol: str
    steps: int = 500000

# =============================================================
# CORE API ENDPOINTS
# =============================================================

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Simple UI for the Command Center"""
    return """
    <html>
        <head>
            <title>PPO Sniper Command Center</title>
            <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap" rel="stylesheet">
            <style>
                body { background-color: #0a0a0a; color: #00ff88; font-family: 'Roboto', sans-serif; padding: 40px; }
                .container { max-width: 900px; margin: 0 auto; background: #111; padding: 30px; border-radius: 15px; border: 1px solid #333; box-shadow: 0 0 20px rgba(0,255,136,0.1); }
                h1 { font-family: 'Orbitron', sans-serif; text-align: center; color: #fff; margin-bottom: 40px; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .card { background: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #333; }
                h2 { color: #00ccff; font-size: 1.2em; border-bottom: 1px solid #444; padding-bottom: 10px; }
                .btn { display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; text-decoration: none; font-weight: bold; border: none; transition: 0.3s; }
                .btn-start { background: #00ff88; color: #000; }
                .btn-stop { background: #ff4444; color: #fff; }
                .active { color: #00ff88; font-weight: bold; }
                .inactive { color: #555; }
                pre { background: #000; padding: 10px; border-radius: 5px; font-size: 0.8em; overflow-x: auto; color: #ccc; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 PPO SNIPER COMMAND CENTER</h1>
                
                <div class="grid">
                    <div class="card">
                        <h2>📡 Live Trader Manager</h2>
                        <p>Status: <span id="trader-status">Checking...</span></p>
                        <button class="btn btn-start" onclick="fetch('/trader/start', {method:'POST'}).then(r=>location.reload())">Launch Suite</button>
                        <button class="btn btn-stop" onclick="fetch('/trader/stop', {method:'POST'}).then(r=>location.reload())">Kill Processes</button>
                    </div>
                    
                    <div class="card">
                        <h2>🧪 Pipeline & Dash</h2>
                        <button class="btn btn-start" onclick="fetch('/pipeline/retrain', {method:'POST'})">Run Retraining Duel</button>
                        <a href="/dashboard" target="_blank" class="btn btn-start" style="background:#00ccff">View Market Intelligence</a>
                    </div>
                </div>

                <div class="card" style="margin-top:20px;">
                    <h2>🖥️ System Console (Last 10 Logs)</h2>
                    <pre id="log-console">Awaiting data...</pre>
                </div>
            </div>
            <script>
                function updateStatus() {
                    fetch('/status').then(r => r.json()).then(data => {
                        document.getElementById('trader-status').innerText = data.processes.trader ? 'ACTIVE' : 'IDLE';
                        document.getElementById('trader-status').className = data.processes.trader ? 'active' : 'inactive';
                    });
                }
                setInterval(updateStatus, 5000);
                updateStatus();
            </script>
        </body>
    </html>
    """

@app.get("/status")
async def get_status():
    """Connectivity and process status"""
    mt5_active = mt5.initialize()
    account = mt5.account_info() if mt5_active else None
    
    # Check if processes are actually running
    proc_status = {}
    for key, proc in processes.items():
        if proc and proc.poll() is None:
            proc_status[key] = True
        else:
            proc_status[key] = False
            processes[key] = None # Reset if dead
            
    return {
        "timestamp": datetime.now().isoformat(),
        "mt5_connected": mt5_active,
        "equity": account.equity if account else 0,
        "processes": proc_status
    }

@app.post("/trader/start")
async def start_trader():
    if processes["trader"] and processes["trader"].poll() is None:
        raise HTTPException(status_code=400, detail="Trader already running")
    
    # Launch Multi-Instance Trader
    cmd = [sys.executable, "multi_instance_trader.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes["trader"] = proc
    return {"message": "Trading Suite Launched"}

@app.post("/trader/stop")
async def stop_all():
    count = 0
    for key in processes:
        if processes[key]:
            processes[key].terminate()
            processes[key] = None
            count += 1
    return {"message": f"Terminated {count} processes"}

@app.post("/pipeline/retrain")
async def run_pipeline(background_tasks: BackgroundTasks):
    if processes["pipeline"] and processes["pipeline"].poll() is None:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    def run():
        subprocess.run([sys.executable, "retrain_pipeline.py"])
        
    background_tasks.add_task(run)
    return {"message": "Retraining Pipeline started in background"}

@app.post("/experts/train")
async def train_expert(req: TrainRequest, background_tasks: BackgroundTasks):
    def run():
        subprocess.run([sys.executable, "universal_trainer.py", req.symbol, str(req.steps)])
        
    background_tasks.add_task(run)
    return {"message": f"Training Expert for {req.symbol} started"}

@app.get("/dashboard")
async def get_dashboard():
    """Serve the latest HTML report"""
    # Trigger dashboard generation first
    subprocess.run([sys.executable, "dashboard.py"])
    path = "results/live_dashboard.html"
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Dashboard not yet generated"}

if __name__ == "__main__":
    import uvicorn
    print("🛰️ PPO Sniper Command Center launching on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
