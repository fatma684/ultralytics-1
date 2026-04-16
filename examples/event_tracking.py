from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from region_utils import RegionCounter, create_demo_regions

Coordinate = tuple[float, float]
Line = tuple[Coordinate, Coordinate]


def parse_sources(values: list[str]) -> list[str]:
    sources: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                sources.append(item)
    return sources if sources else ["0"]


def create_default_line(frame: np.ndarray, line_position: float = 0.6) -> Line:
    h, w = frame.shape[:2]
    return ((int(w * 0.1), int(h * line_position)), (int(w * 0.9), int(h * line_position)))


def line_crossed(prev: Coordinate | None, curr: Coordinate, line: Line) -> str | None:
    if prev is None:
        return None

    (x0, y0), (x1, y1) = line
    if y0 == y1:
        line_y = y0
        if (prev[1] - line_y) * (curr[1] - line_y) < 0:
            x_cross = prev[0] + (curr[0] - prev[0]) * (line_y - prev[1]) / (curr[1] - prev[1])
            if min(x0, x1) <= x_cross <= max(x0, x1):
                return "in" if curr[1] > prev[1] else "out"
    elif x0 == x1:
        line_x = x0
        if (prev[0] - line_x) * (curr[0] - line_x) < 0:
            y_cross = prev[1] + (curr[1] - prev[1]) * (line_x - prev[0]) / (curr[0] - prev[0])
            if min(y0, y1) <= y_cross <= max(y0, y1):
                return "in" if curr[0] > prev[0] else "out"

    return None


def draw_text(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


@dataclass
class EventCamera:
    source: str
    weights: str
    tracker: str
    classes: list[int]
    conf: float
    iou: float
    line_position: float = 0.6
    heatmap_alpha: float = 0.45
    enable_regions: bool = False
    name: str = field(init=False)

    cap: cv2.VideoCapture = field(init=False)
    model: YOLO = field(init=False)
    heatmap: np.ndarray | None = field(init=False, default=None)
    line: Line | None = field(init=False, default=None)
    track_history: dict[int, Coordinate] = field(init=False, default_factory=dict)
    unique_ids: set[int] = field(init=False, default_factory=set)
    in_count: int = field(init=False, default=0)
    out_count: int = field(init=False, default=0)
    current_ids: set[int] = field(init=False, default_factory=set)
    colormap_id: int = field(init=False, default=cv2.COLORMAP_JET)
    region_counter: RegionCounter | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.name = f"Cam {self.source}"
        self.cap = cv2.VideoCapture(int(self.source)) if self.source.isdigit() else cv2.VideoCapture(self.source)
        self.model = YOLO(self.weights)
        self.colormap_id = cv2.COLORMAP_JET

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()

    def initialize_frame(self, frame: np.ndarray) -> None:
        self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        self.line = create_default_line(frame, line_position=self.line_position)
        if self.enable_regions and self.region_counter is None:
            self.region_counter = create_demo_regions(frame.shape[0], frame.shape[1])

    def heatmap_pulse(self, box: list[float]) -> None:
        if self.heatmap is None:
            return

        x0, y0, x1, y1 = map(int, map(round, box))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.heatmap.shape[1] - 1, x1), min(self.heatmap.shape[0] - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return

        center_x = int((x0 + x1) / 2)
        center_y = int((y0 + y1) / 2)
        radius = max(4, min(x1 - x0, y1 - y0) // 2)
        mask = np.zeros_like(self.heatmap, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        self.heatmap += mask.astype(np.float32) * 2.0

    def overlay_heatmap(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap is None or np.count_nonzero(self.heatmap) == 0:
            return frame

        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_norm, self.colormap_id)
        return cv2.addWeighted(frame, 1.0 - self.heatmap_alpha, color_heatmap, self.heatmap_alpha, 0)

    def draw_region(self, frame: np.ndarray) -> None:
        if self.line is None:
            return
        cv2.line(frame, self.line[0], self.line[1], (0, 255, 255), 3, cv2.LINE_AA)

    def annotate(self, frame: np.ndarray) -> None:
        draw_text(frame, f"Source: {self.source}", (10, 25))
        draw_text(frame, f"Unique IDs: {len(self.unique_ids)}", (10, 55))
        draw_text(frame, f"Current crowd: {len(self.current_ids)}", (10, 85))
        draw_text(frame, f"IN: {self.in_count}  OUT: {self.out_count}", (10, 115))
        if self.region_counter:
            y_offset = 145
            for region_name, count in self.region_counter.get_summary().items():
                draw_text(frame, f"{region_name}: {count}", (10, y_offset), color=(150, 200, 150))
                y_offset += 30
            draw_text(frame, "ESC = quit", (10, frame.shape[0] - 20), color=(200, 200, 200))
        else:
            draw_text(frame, "ESC = quit", (10, frame.shape[0] - 20), color=(200, 200, 200))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap is None or self.line is None:
            self.initialize_frame(frame)

        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        if hasattr(results.boxes, "id") and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([], dtype=int)

        self.current_ids = set(ids.tolist())
        self.unique_ids.update(self.current_ids)

        annotated = results.plot()

        if len(ids) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            for box, track_id in zip(boxes, ids):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                prev_center = self.track_history.get(track_id)
                direction = line_crossed(prev_center, center, self.line)
                if direction == "in":
                    self.in_count += 1
                elif direction == "out":
                    self.out_count += 1
                self.track_history[track_id] = center
                self.heatmap_pulse(box)

                # Update region counts
                if self.region_counter:
                    self.region_counter.update(track_id, center)

        annotated = self.overlay_heatmap(annotated)
        self.draw_region(annotated)

        # Draw regions if enabled
        if self.region_counter:
            self.region_counter.draw_all(annotated, line_width=2)
            self.region_counter.reset_tracked_ids()

        self.annotate(annotated)

        return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO event tracking demo: crowd counter, automatic entry/exit, heatmap, multi-camera")
    parser.add_argument("--weights", "-w", default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument(
        "--sources",
        "-s",
        nargs="+",
        default=["0"],
        help="Video sources or camera indices (e.g. 0 1 or video.mp4)",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config to use")
    parser.add_argument("--classes", "-c", nargs="*", type=int, default=[0], help="Classes to track (default person class 0)")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for tracker")
    parser.add_argument("--line-position", type=float, default=0.6, help="Position of entry/exit line as fraction of frame height")
    parser.add_argument(
        "--colormap",
        choices=["JET", "TURBO", "HOT", "COOL", "SPRING", "SUMMER", "AUTUMN", "WINTER", "RAINBOW", "OCEAN", "VIRIDIS"],
        default="JET",
        help="Heatmap colormap",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Enable multi-region counting (3 zones: LEFT, CENTER, RIGHT)",
    )
    args = parser.parse_args()

    # Map colormap names to OpenCV constants
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "HOT": cv2.COLORMAP_HOT,
        "COOL": cv2.COLORMAP_COOL,
        "SPRING": cv2.COLORMAP_SPRING,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }

    sources = parse_sources(args.sources)
    cameras: list[EventCamera] = []

    for source in sources:
        camera = EventCamera(
            source=source,
            weights=args.weights,
            tracker=args.tracker,
            classes=args.classes,
            conf=args.conf,
            iou=args.iou,
            line_position=args.line_position,
            enable_regions=args.regions,
        )
        camera.colormap_id = colormap_dict.get(args.colormap, cv2.COLORMAP_JET)
        if camera.is_open():
            cameras.append(camera)
        else:
            print(f"Impossible d'ouvrir la source: {source}")

    if not cameras:
        raise SystemExit("Aucune source valide trouvée. Vérifiez les indices de caméra ou les chemins de fichier.")

    print("Démarrage du suivi événementiel pour:")
    for camera in cameras:
        print(f" - {camera.source}")

    while True:
        any_active = False
        for camera in cameras:
            if not camera.is_open():
                continue

            any_active = True
            ret, frame = camera.cap.read()
            if not ret or frame is None:
                continue

            annotated = camera.process_frame(frame)
            cv2.imshow(camera.name, annotated)

        if not any_active:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()

    print("\nRésumé final :")
    for camera in cameras:
        print(
            f"{camera.name}  | unique_ids={len(camera.unique_ids)} | current={len(camera.current_ids)} | in={camera.in_count} | out={camera.out_count}"
        )


if __name__ == "__main__":
    main()
