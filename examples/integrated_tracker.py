"""Integrated Event Tracking Pipeline - Connect cameras to event service and API."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from event_service import DetectionEvent, EventService, EventType
from region_utils import RegionCounter, create_demo_regions

Coordinate = tuple[float, float]
Line = tuple[Coordinate, Coordinate]


class IntegratedEventTracker:
    """Integrated tracking system with event service."""

    def __init__(self, camera_source: str | int, weights: str = "yolov8n.pt", camera_id: str | None = None) -> None:
        """Initialize integrated tracker."""
        self.camera_source = camera_source
        self.camera_id = camera_id or f"camera_{camera_source}"
        self.model = YOLO(weights)
        self.event_service: EventService | None = None

        self.cap = cv2.VideoCapture(int(camera_source)) if isinstance(camera_source, str) and camera_source.isdigit() else cv2.VideoCapture(camera_source)

        self.track_history: dict[int, Coordinate] = {}
        self.unique_ids: set[int] = set()
        self.detected_objects: list[dict[str, Any]] = []

        self.line: Line | None = None
        self.region_counter: RegionCounter | None = None
        self.in_count = 0
        self.out_count = 0

    def set_event_service(self, service: EventService) -> None:
        """Set the event service for recording events."""
        self.event_service = service

    def initialize_frame(self, frame: np.ndarray) -> None:
        """Initialize frame-based settings."""
        h, w = frame.shape[:2]
        self.line = ((int(w * 0.1), int(h * 0.6)), (int(w * 0.9), int(h * 0.6)))
        self.region_counter = create_demo_regions(h, w)

    def line_crossed(self, prev: Coordinate | None, curr: Coordinate) -> str | None:
        """Check if object crossed the line."""
        if prev is None or self.line is None:
            return None

        (x0, y0), (x1, y1) = self.line
        if y0 == y1:
            line_y = y0
            if (prev[1] - line_y) * (curr[1] - line_y) < 0:
                x_cross = prev[0] + (curr[0] - prev[0]) * (line_y - prev[1]) / (curr[1] - prev[1])
                if min(x0, x1) <= x_cross <= max(x0, x1):
                    return "in" if curr[1] > prev[1] else "out"
        return None

    def record_event(self, event_type: EventType, track_id: int, class_name: str, confidence: float, box: list[float], region_name: str | None = None) -> None:
        """Record event to service."""
        if self.event_service is None:
            return

        x_min, y_min, x_max, y_max = box
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2

        event = DetectionEvent(
            timestamp=__import__("datetime").datetime.now(),
            camera_id=self.camera_id,
            track_id=track_id,
            event_type=event_type,
            class_name=class_name,
            confidence=confidence,
            x=x,
            y=y,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            region_name=region_name,
        )
        self.event_service.record_event(event)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        if self.line is None:
            self.initialize_frame(frame)

        # Run tracking
        results = self.model.track(frame, persist=True, conf=0.3, verbose=False)[0]

        if hasattr(results.boxes, "id") and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([], dtype=int)

        self.detected_objects = []
        self.unique_ids.update(ids.tolist())

        annotated = results.plot()

        if len(ids) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_idx in zip(boxes, ids, confs, clss):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                prev_center = self.track_history.get(track_id)

                # Check line crossing
                direction = self.line_crossed(prev_center, center)
                if direction == "in":
                    self.in_count += 1
                    self.record_event(EventType.ENTRY, track_id, self.model.names[cls_idx], float(conf), box.tolist())
                elif direction == "out":
                    self.out_count += 1
                    self.record_event(EventType.EXIT, track_id, self.model.names[cls_idx], float(conf), box.tolist())

                # Check region
                if self.region_counter:
                    for region in self.region_counter.regions:
                        if region.contains(center) and track_id not in region.tracked_ids:
                            self.record_event(EventType.REGION_ENTER, track_id, self.model.names[cls_idx], float(conf), box.tolist(), region.name)
                            region.tracked_ids.add(track_id)

                self.track_history[track_id] = center
                self.detected_objects.append({
                    "track_id": track_id,
                    "class": self.model.names[cls_idx],
                    "confidence": float(conf),
                    "box": box.tolist(),
                    "center": center,
                })

        # Draw annotations
        if self.line is not None:
            cv2.line(annotated, self.line[0], self.line[1], (0, 255, 255), 3)
        if self.region_counter:
            self.region_counter.draw_all(annotated)
            self.region_counter.reset_tracked_ids()

        # Add text
        cv2.putText(annotated, f"IN: {self.in_count} OUT: {self.out_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Unique IDs: {len(self.unique_ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def run(self) -> None:
        """Run the tracker."""
        print(f"Starting tracker for camera: {self.camera_id}")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame)
                cv2.imshow(f"Tracker - {self.camera_id}", annotated)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "camera_id": self.camera_id,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "unique_ids": len(self.unique_ids),
            "detected_objects": self.detected_objects,
        }


def run_tracker_with_service(camera_source: str | int, event_service: EventService, camera_id: str | None = None) -> None:
    """Run tracker in a thread with event service."""
    tracker = IntegratedEventTracker(camera_source, camera_id=camera_id)
    tracker.set_event_service(event_service)
    tracker.run()


if __name__ == "__main__":
    # Example usage
    service = EventService()
    tracker = IntegratedEventTracker(0, camera_id="cam_0")
    tracker.set_event_service(service)
    tracker.run()
