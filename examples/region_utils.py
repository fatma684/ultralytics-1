# Ultralytics Region Counter Utilities

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

Coordinate = tuple[float, float]


@dataclass
class Region:
    """Represents a polygonal region for object counting."""

    name: str
    points: list[Coordinate]
    color: tuple[int, int, int]
    count: int = field(default=0)
    tracked_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Convert to numpy array for faster polygon operations."""
        self.points_array = np.array(self.points, dtype=np.int32)

    def contains(self, point: Coordinate) -> bool:
        """Check if a point is inside the polygon region."""
        return cv2.pointPolygonTest(self.points_array, point, False) >= 0

    def draw(self, frame: np.ndarray, line_width: int = 2, font_scale: float = 0.6) -> None:
        """Draw the region on the frame."""
        cv2.polylines(frame, [self.points_array], True, self.color, line_width)
        centroid_x = int(np.mean(self.points_array[:, 0]))
        centroid_y = int(np.mean(self.points_array[:, 1]))
        cv2.putText(
            frame,
            f"{self.name}: {self.count}",
            (centroid_x - 30, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.color,
            2,
        )

    def reset_count(self) -> None:
        """Reset the count for the region."""
        self.count = 0


class RegionCounter:
    """Manages multiple polygonal regions for object counting."""

    def __init__(self) -> None:
        """Initialize the region counter."""
        self.regions: list[Region] = []

    def add_region(
        self,
        name: str,
        points: list[Coordinate],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> Region:
        """Add a new region."""
        region = Region(name=name, points=points, color=color)
        self.regions.append(region)
        return region

    def update(self, track_id: int, center: Coordinate) -> None:
        """Update region counts with a tracked object."""
        for region in self.regions:
            if region.contains(center):
                if track_id not in region.tracked_ids:
                    region.count += 1
                    region.tracked_ids.add(track_id)

    def reset_tracked_ids(self) -> None:
        """Reset tracked IDs for next frame."""
        for region in self.regions:
            region.tracked_ids.clear()

    def draw_all(self, frame: np.ndarray, line_width: int = 2) -> None:
        """Draw all regions on the frame."""
        for region in self.regions:
            region.draw(frame, line_width=line_width)

    def get_summary(self) -> dict[str, int]:
        """Get a summary of all region counts."""
        return {region.name: region.count for region in self.regions}


def create_demo_regions(frame_height: int, frame_width: int) -> RegionCounter:
    """Create demo regions for a frame."""
    counter = RegionCounter()

    # Left zone
    counter.add_region(
        "Zone LEFT",
        [(50, 50), (frame_width // 3, 50), (frame_width // 3, frame_height - 50), (50, frame_height - 50)],
        color=(255, 0, 0),
    )

    # Center zone
    counter.add_region(
        "Zone CENTER",
        [(frame_width // 3 + 10, 50), (2 * frame_width // 3 - 10, 50), (2 * frame_width // 3 - 10, frame_height - 50), (frame_width // 3 + 10, frame_height - 50)],
        color=(0, 255, 0),
    )

    # Right zone
    counter.add_region(
        "Zone RIGHT",
        [(2 * frame_width // 3, 50), (frame_width - 50, 50), (frame_width - 50, frame_height - 50), (2 * frame_width // 3, frame_height - 50)],
        color=(0, 0, 255),
    )

    return counter
