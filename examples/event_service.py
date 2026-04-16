"""Event Service - Core event tracking and storage logic."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class EventType(str, Enum):
    """Event types for tracking system."""

    ENTRY = "entry"
    EXIT = "exit"
    REGION_ENTER = "region_enter"
    REGION_EXIT = "region_exit"
    DETECTION = "detection"


@dataclass
class DetectionEvent:
    """Represents a single detection event."""

    timestamp: datetime
    camera_id: str
    track_id: int
    event_type: EventType
    class_name: str
    confidence: float
    x: float
    y: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    region_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["event_type"] = self.event_type.value
        return d


@dataclass
class CameraStats:
    """Statistics for a camera."""

    camera_id: str
    total_tracks: int
    entry_count: int
    exit_count: int
    current_crowd: int
    unique_ids: set[int] = field(default_factory=set)
    region_counts: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "camera_id": self.camera_id,
            "total_tracks": self.total_tracks,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "current_crowd": self.current_crowd,
            "unique_ids_count": len(self.unique_ids),
            "region_counts": self.region_counts,
            "last_updated": self.last_updated.isoformat(),
        }


class EventService:
    """Service for managing detection events and statistics."""

    def __init__(self) -> None:
        """Initialize the event service."""
        self.events: list[DetectionEvent] = []
        self.stats: dict[str, CameraStats] = {}
        self.max_events = 10000  # Keep last N events in memory

    def record_event(self, event: DetectionEvent) -> None:
        """Record a detection event."""
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

        # Update statistics
        if event.camera_id not in self.stats:
            self.stats[event.camera_id] = CameraStats(
                camera_id=event.camera_id,
                total_tracks=0,
                entry_count=0,
                exit_count=0,
                current_crowd=0,
            )

        stats = self.stats[event.camera_id]
        stats.unique_ids.add(event.track_id)
        stats.last_updated = datetime.now()

        if event.event_type == EventType.ENTRY:
            stats.entry_count += 1
            stats.current_crowd += 1
        elif event.event_type == EventType.EXIT:
            stats.exit_count += 1
            stats.current_crowd = max(0, stats.current_crowd - 1)
        elif event.event_type == EventType.REGION_ENTER:
            if event.region_name:
                stats.region_counts[event.region_name] = stats.region_counts.get(event.region_name, 0) + 1

    def get_events(self, camera_id: str | None = None, event_type: EventType | None = None, limit: int = 100) -> list[DetectionEvent]:
        """Get events with optional filtering."""
        events = self.events
        if camera_id:
            events = [e for e in events if e.camera_id == camera_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_stats(self, camera_id: str | None = None) -> dict[str, Any]:
        """Get statistics for camera(s)."""
        if camera_id:
            stats = self.stats.get(camera_id)
            return stats.to_dict() if stats else {}
        return {cid: stats.to_dict() for cid, stats in self.stats.items()}

    def clear_events(self) -> None:
        """Clear all events."""
        self.events.clear()
