"""
FastAPI Server - REST API + WebSocket for real-time match statistics.

Endpoints (from SDD):
POST /matches           - Create match
POST /matches/{id}/start - Start processing
GET  /matches/{id}/stats - Current stats
GET  /matches/{id}/events - Event list
GET  /matches/{id}/state  - Current match state
WS   /matches/{id}/live   - Real-time push
POST /events/{id}/confirm - Confirm event
POST /events/{id}/reject  - Reject event
POST /matches/{id}/manual-event - Manual event entry
"""
import asyncio
import json
import os
import sys
from typing import Optional, Dict, List
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.match_engine import MatchEngine

app = FastAPI(
    title="Football Intelligence API",
    description="Real-time football match statistics from video analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
matches: Dict[str, MatchEngine] = {}
active_websockets: Dict[str, List[WebSocket]] = {}
processing_tasks: Dict[str, asyncio.Task] = {}


class MatchCreate(BaseModel):
    match_id: str
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"


class MatchStart(BaseModel):
    video_path: str
    output_path: Optional[str] = None
    max_frames: Optional[int] = None


class ManualEvent(BaseModel):
    event_type: str  # goal, yellow_card, red_card, corner
    team_id: str  # A or B
    timestamp_ms: float = 0


class EventDecision(BaseModel):
    decision: str  # confirmed or rejected
    reviewer: str = "operator"


@app.get("/")
async def root():
    return {
        "name": "Football Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "POST /matches": "Create a new match",
            "POST /matches/{id}/start": "Start video processing",
            "GET /matches/{id}/stats": "Get current statistics",
            "GET /matches/{id}/events": "Get all events",
            "GET /matches/{id}/state": "Get current match state",
            "WS /matches/{id}/live": "Real-time WebSocket updates",
            "POST /matches/{id}/manual-event": "Add manual event",
        }
    }


@app.post("/matches")
async def create_match(req: MatchCreate):
    """Create a new match."""
    if req.match_id in matches:
        raise HTTPException(400, "Match already exists")

    engine = MatchEngine()
    engine.stats.stats.match_id = req.match_id
    matches[req.match_id] = engine
    active_websockets[req.match_id] = []

    return {
        "match_id": req.match_id,
        "status": "created",
        "team_a": req.team_a_name,
        "team_b": req.team_b_name
    }


@app.post("/matches/{match_id}/start")
async def start_match(match_id: str, req: MatchStart, bg: BackgroundTasks):
    """Start processing a match video."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")

    engine = matches[match_id]

    async def process_video_async():
        try:
            def progress_cb(frame_num, total, result):
                # Broadcast to WebSocket clients
                asyncio.run_coroutine_threadsafe(
                    broadcast_update(match_id, result),
                    asyncio.get_event_loop()
                )

            final = engine.process_video(
                req.video_path,
                output_path=req.output_path,
                progress_callback=progress_cb,
                max_frames=req.max_frames
            )
            return final
        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")

    task = asyncio.create_task(process_video_async())
    processing_tasks[match_id] = task

    return {
        "match_id": match_id,
        "status": "processing",
        "video_path": req.video_path
    }


@app.get("/matches/{match_id}/stats")
async def get_stats(match_id: str):
    """Get current match statistics."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")
    return matches[match_id].stats.snapshot()


@app.get("/matches/{match_id}/events")
async def get_events(match_id: str):
    """Get all events (confirmed + pending)."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")
    engine = matches[match_id]
    events = engine.fusion_engine.get_all_events()
    return {
        "match_id": match_id,
        "events": [
            {
                "type": e.event_type.value,
                "team": e.team_id,
                "time_ms": e.start_ms,
                "confidence": e.confidence,
                "status": e.status,
                "evidence": e.evidence
            }
            for e in events
        ]
    }


@app.get("/matches/{match_id}/state")
async def get_state(match_id: str):
    """Get current match state."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")
    engine = matches[match_id]
    return {
        "match_id": match_id,
        "frame_count": engine.frame_count,
        "current_time_ms": engine.current_time_ms,
        "stats": engine.stats.snapshot(),
        "pending_events": [
            {
                "type": e.event_type.value,
                "team": e.team_id,
                "time_ms": e.start_ms,
                "confidence": e.confidence
            }
            for e in engine.fusion_engine.get_pending()
        ]
    }


@app.post("/matches/{match_id}/manual-event")
async def add_manual_event(match_id: str, req: ManualEvent):
    """Manually add an event (goal, card, corner)."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")

    from core.events.detector import EventType
    engine = matches[match_id]

    type_map = {
        "goal": EventType.GOAL,
        "yellow_card": EventType.YELLOW_CARD,
        "red_card": EventType.RED_CARD,
        "corner": EventType.CORNER
    }

    event_type = type_map.get(req.event_type)
    if not event_type:
        raise HTTPException(400, f"Unknown event type: {req.event_type}")

    event = engine.fusion_engine.manual_add_event(event_type, req.team_id, req.timestamp_ms)
    engine.stats.update_from_events([event])

    return {
        "status": "added",
        "event": {
            "type": event.event_type.value,
            "team": event.team_id,
            "time_ms": event.start_ms
        },
        "stats": engine.stats.snapshot()
    }


@app.post("/events/{event_id}/confirm")
async def confirm_event(event_id: str, match_id: str, req: EventDecision):
    """Confirm a pending event."""
    if match_id not in matches:
        raise HTTPException(404, "Match not found")
    engine = matches[match_id]
    success = engine.fusion_engine.manual_confirm(event_id, req.decision, req.reviewer)
    if not success:
        raise HTTPException(404, "Event not found")
    return {"status": req.decision}


@app.websocket("/matches/{match_id}/live")
async def websocket_live(ws: WebSocket, match_id: str):
    """WebSocket endpoint for real-time stats push."""
    await ws.accept()

    if match_id not in active_websockets:
        active_websockets[match_id] = []
    active_websockets[match_id].append(ws)

    try:
        while True:
            # Keep connection alive, send current stats periodically
            if match_id in matches:
                stats = matches[match_id].stats.snapshot()
                await ws.send_json(stats)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_websockets[match_id].remove(ws)


async def broadcast_update(match_id: str, result: dict):
    """Broadcast update to all WebSocket clients for a match."""
    if match_id in active_websockets:
        for ws in active_websockets[match_id]:
            try:
                await ws.send_json(result.get("stats", {}))
            except Exception:
                pass


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
