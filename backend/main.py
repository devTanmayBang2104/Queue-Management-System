"""
FastAPI Application - Queue Management System Backend
Provides REST API endpoints and WebSocket for real-time video streaming.

Endpoints:
    GET  /api/analytics   - Current analytics data
    GET  /api/prediction  - Queue wait time prediction
    GET  /api/alerts      - Recent alerts
    GET  /api/heatmap     - Current heatmap image
    GET  /api/history     - Historical queue data
    GET  /api/config      - Current configuration
    POST /api/config      - Update configuration
    POST /api/start       - Start pipeline with video source
    POST /api/stop        - Stop pipeline
    POST /api/upload      - Upload video file
    WS   /ws              - WebSocket for live data stream
"""

import os
import sys
import json
import asyncio
import time
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import AppConfig, ROIConfig, QueueConfig
from backend.pipeline import CVPipeline
from backend.database import get_db, QueueStat, AlertHistory, init_db

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QueueVision AI",
    description="AI-Powered Queue Management System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = AppConfig()
pipeline = CVPipeline(config)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    print("[Server] QueueVision AI Backend ready")
    print("[Server] Waiting for pipeline start command...")


@app.on_event("shutdown")
async def shutdown():
    pipeline.stop()
    print("[Server] Shutdown complete")


# ── Request Models ─────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    video_source: Optional[str] = "0"

class ConfigUpdate(BaseModel):
    max_queue_length_alert: Optional[int] = None
    max_wait_time_alert: Optional[float] = None
    crowd_spike_threshold: Optional[int] = None
    velocity_threshold: Optional[float] = None
    rois: Optional[List[dict]] = None


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "QueueVision AI", "version": "1.0.0"}


@app.post("/api/start")
async def start_pipeline(req: StartRequest):
    """Start the CV pipeline with a video source."""
    if pipeline.state.running:
        return {"status": "already_running"}
    loop = asyncio.get_event_loop()
    pipeline.start(video_source=req.video_source, loop=loop)
    return {"status": "started", "source": req.video_source}


@app.post("/api/stop")
async def stop_pipeline():
    """Stop the CV pipeline."""
    pipeline.stop()
    return {"status": "stopped"}


@app.get("/api/status")
async def get_status():
    """Get pipeline status."""
    return {
        "running": pipeline.state.running,
        "source": pipeline.state._video_source,
    }


@app.get("/api/analytics")
async def get_analytics():
    """Get current queue analytics."""
    stats = pipeline.state.get_stats()
    snapshot = pipeline.state.get_snapshot()
    return {
        "stats": stats,
        "persons": snapshot.get("persons", []),
        "timestamp": time.time(),
    }


@app.get("/api/prediction")
async def get_prediction():
    """Get queue wait time prediction."""
    stats = pipeline.state.get_stats()
    return {
        "predicted_wait_time": stats.get("predicted_wait_time", 0),
        "queue_count": stats.get("queue_count", 0),
        "arrival_rate": stats.get("arrival_rate", 0),
        "service_rate": stats.get("service_rate", 0),
    }


@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts from DB or memory fallback."""
    db = get_db()
    try:
        alerts_db = db.query(AlertHistory).order_by(AlertHistory.timestamp.desc()).limit(limit).all()
        alerts = []
        for a in alerts_db:
            alerts.append({
                "type": a.alert_type,
                "message": a.message,
                "severity": a.severity,
                "threshold": a.threshold_value,
                "actual": a.actual_value,
                "timestamp": a.timestamp.timestamp()
            })
        if alerts:
            return {"alerts": alerts}
    except Exception as e:
        print(f"[DB] Error fetching alerts: {e}")
    finally:
        db.close()

    alerts = pipeline.alert_manager.get_history(limit) if pipeline.alert_manager else []
    return {"alerts": alerts}


@app.get("/api/heatmap")
async def get_heatmap():
    """Get current heatmap as base64 JPEG."""
    hm = pipeline.state.get_heatmap()
    if hm is None:
        return {"heatmap": None}
    return {"heatmap": hm}


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get historical queue stats from DB or memory fallback."""
    db = get_db()
    try:
        stats = db.query(QueueStat).order_by(QueueStat.timestamp.desc()).limit(limit).all()
        history = []
        for s in reversed(stats):
            history.append({
                "timestamp": s.timestamp.timestamp(),
                "stats": {
                    "queue_count": s.queue_length,
                    "avg_wait_time": s.avg_wait_time,
                    "max_wait_time": s.max_wait_time,
                    "predicted_wait_time": s.predicted_wait_time,
                    "arrival_rate": s.arrival_rate,
                    "service_rate": s.service_rate,
                    "abandonment_rate": s.abandonment_rate,
                }
            })
        if history:
            return {"history": history}
    except Exception as e:
        print(f"[DB] Error fetching history: {e}")
    finally:
        db.close()

    history = pipeline.state.get_history()
    return {"history": history[-limit:]}


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return config.model_dump()


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration (thresholds, ROIs)."""
    if update.max_queue_length_alert is not None:
        config.queue.max_queue_length_alert = update.max_queue_length_alert
    if update.max_wait_time_alert is not None:
        config.queue.max_wait_time_alert = update.max_wait_time_alert
    if update.crowd_spike_threshold is not None:
        config.queue.crowd_spike_threshold = update.crowd_spike_threshold

    if update.rois is not None:
        config.rois = [ROIConfig(**r) for r in update.rois]
        if pipeline.queue_detector:
            pipeline.queue_detector.set_rois(config.rois)

    if pipeline.alert_manager:
        pipeline.alert_manager.update_thresholds(
            max_queue=config.queue.max_queue_length_alert,
            max_wait=config.queue.max_wait_time_alert,
            crowd_thresh=config.queue.crowd_spike_threshold,
        )

    return {"status": "updated", "config": config.model_dump()}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}")

    save_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "uploaded", "path": save_path, "filename": file.filename}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live data streaming."""
    await ws.accept()
    pipeline.add_ws_client(ws)
    print(f"[WS] Client connected. Total: {len(pipeline.ws_clients)}")

    try:
        while True:
            # Keep connection alive, handle client messages
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Handle client commands
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await ws.send_json({"type": "heartbeat", "timestamp": time.time()})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        pipeline.remove_ws_client(ws)
        print(f"[WS] Client disconnected. Total: {len(pipeline.ws_clients)}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
