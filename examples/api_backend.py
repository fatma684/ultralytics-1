"""FastAPI Backend - REST API for event tracking system."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from event_service import EventService, EventType, DetectionEvent

app = FastAPI(title="Event Tracking API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize event service
event_service = EventService()


# Pydantic models for API
class EventResponse(BaseModel):
    """Response model for events."""

    timestamp: str
    camera_id: str
    track_id: int
    event_type: str
    class_name: str
    confidence: float
    x: float
    y: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    region_name: str | None = None
    metadata: dict[str, Any]


class CameraStatsResponse(BaseModel):
    """Response model for camera statistics."""

    camera_id: str
    total_tracks: int
    entry_count: int
    exit_count: int
    current_crowd: int
    unique_ids_count: int
    region_counts: dict[str, int]
    last_updated: str


class EventCreateRequest(BaseModel):
    """Request model for creating events."""

    camera_id: str
    track_id: int
    event_type: str
    class_name: str
    confidence: float
    x: float
    y: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    region_name: str | None = None
    metadata: dict[str, Any] = {}


# Routes
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/events")
async def create_event(request: EventCreateRequest) -> EventResponse:
    """Create a new event."""
    try:
        event_type = EventType[request.event_type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {request.event_type}")

    event = DetectionEvent(
        timestamp=datetime.now(),
        camera_id=request.camera_id,
        track_id=request.track_id,
        event_type=event_type,
        class_name=request.class_name,
        confidence=request.confidence,
        x=request.x,
        y=request.y,
        x_min=request.x_min,
        y_min=request.y_min,
        x_max=request.x_max,
        y_max=request.y_max,
        region_name=request.region_name,
        metadata=request.metadata,
    )
    event_service.record_event(event)
    return EventResponse(**event.to_dict())


@app.get("/events")
async def get_events(camera_id: str | None = None, event_type: str | None = None, limit: int = 100) -> list[EventResponse]:
    """Get events with optional filtering."""
    try:
        et = EventType[event_type.upper()] if event_type else None
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")

    events = event_service.get_events(camera_id=camera_id, event_type=et, limit=limit)
    return [EventResponse(**e.to_dict()) for e in events]


@app.get("/stats")
async def get_stats(camera_id: str | None = None) -> dict[str, Any]:
    """Get statistics for camera(s)."""
    return event_service.get_stats(camera_id=camera_id)


@app.get("/stats/{camera_id}")
async def get_camera_stats(camera_id: str) -> CameraStatsResponse:
    """Get statistics for a specific camera."""
    stats = event_service.get_stats(camera_id=camera_id)
    if not stats:
        raise HTTPException(status_code=404, detail=f"No stats found for camera {camera_id}")
    return CameraStatsResponse(**stats)


@app.delete("/events")
async def clear_events() -> dict[str, str]:
    """Clear all events."""
    event_service.clear_events()
    return {"message": "Events cleared"}


@app.get("/summary")
async def get_summary() -> dict[str, Any]:
    """Get a summary of all cameras."""
    stats = event_service.get_stats()
    total_events = len(event_service.events)
    total_cameras = len(stats)
    total_entries = sum(s["entry_count"] for s in stats.values())
    total_exits = sum(s["exit_count"] for s in stats.values())
    total_crowd = sum(s["current_crowd"] for s in stats.values())

    return {
        "total_events": total_events,
        "total_cameras": total_cameras,
        "total_entries": total_entries,
        "total_exits": total_exits,
        "total_crowd": total_crowd,
        "cameras": stats,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
