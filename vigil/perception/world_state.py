"""Core world-state model: scene graph built from tracked detections.

This module contains the pure data model for maintaining a spatial-temporal
scene graph.  It has **no dependency on any detection backend** (YOLO,
ultralytics, etc.) — detections are fed in via
:meth:`WorldState.update_from_detections`.
"""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IOU_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 100
MOVEMENT_THRESHOLD = 8
MOTION_START_CONTINUOUS_FRAME_THRESHOLD = 4
MOTION_STOP_CONTINUOUS_FRAME_THRESHOLD = 4
MAX_EVENTS = 20
DISAPPEARANCE_THRESHOLD = 15
CROP_BUFFER = 20

DIRECTION_MAPPING = {
    -4: "left",
    -3: "bottom left",
    -2: "below",
    -1: "bottom right",
    0: "right",
    1: "upper right",
    2: "above",
    3: "upper left",
    4: "left",
}


# ---------------------------------------------------------------------------
# Lightweight IoU (replaces ultralytics.utils.metrics.bbox_iou)
# ---------------------------------------------------------------------------


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two ``[x1, y1, x2, y2]`` bounding boxes.

    Both *box_a* and *box_b* can be 1-D arrays or tensors of length 4.  The
    function returns a plain Python float so it works identically to the
    ``(bbox_iou(a, b) > threshold).item()`` pattern used in the original code.
    """
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def safe_crop(frame: np.ndarray, xyxy, buffer_px: int = 10) -> np.ndarray:
    """Crop a region from *frame* defined by *xyxy* with a pixel buffer."""
    h, w = frame.shape[:2]

    x1 = int(xyxy[0].item()) - buffer_px
    y1 = int(xyxy[1].item()) - buffer_px
    x2 = int(xyxy[2].item()) + buffer_px
    y2 = int(xyxy[3].item()) + buffer_px

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class Relation:
    """Spatial relation between two tracked objects."""

    def __init__(
        self,
        subject_id: str,
        direction: str,
        near: bool,
        overlapping: bool,
        object_id: str,
        last_updated,
    ):
        self.subject_id = subject_id
        self.direction = direction
        self.near = near
        self.overlapping = overlapping
        self.object_id = object_id
        self.last_updated = last_updated

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject_id,
            "relation": (
                self.direction
                + ", "
                + ("near" if self.near else "far")
                + ", "
                + ("overlapping" if self.overlapping else "not overlapping")
            ),
            "object": self.object_id,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# WorldObject
# ---------------------------------------------------------------------------


class WorldObject:
    """A single tracked object in the scene."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        center,
        confidence: float,
        frame_idx: int,
        xyxy,
    ):
        self.track_id = track_id
        self.type = class_name

        self.xyxy = xyxy
        self.center = center
        self.prev_center = center
        self.velocity = (0.0, 0.0)

        self.visible = True
        self.moving = False
        self.motion_counter = 0
        self.stationary_counter = 0

        self.confidence = confidence
        self.first_seen = frame_idx
        self.last_seen = frame_idx

    def update(self, center, confidence: float, frame_idx: int, xyxy) -> None:
        vx = center[0] - self.center[0]
        vy = center[1] - self.center[1]

        self.xyxy = xyxy
        self.prev_center = self.center
        self.center = center
        self.velocity = (vx, vy)
        speed = math.hypot(vx, vy)

        if speed > MOVEMENT_THRESHOLD:
            self.motion_counter += 1
            self.stationary_counter = 0
        else:
            self.stationary_counter += 1
            self.motion_counter = 0

        if (not self.moving) and (
            self.motion_counter >= MOTION_START_CONTINUOUS_FRAME_THRESHOLD
        ):
            self.moving = True
        elif (self.moving) and (
            self.stationary_counter >= MOTION_STOP_CONTINUOUS_FRAME_THRESHOLD
        ):
            self.moving = False

        self.confidence = confidence
        self.last_seen = frame_idx
        self.visible = True

    def mark_missing(self) -> None:
        self.visible = False

    def to_dict(self) -> dict:
        return {
            "id": self.track_id,
            "type": self.type,
            "position": {"x": self.center[0], "y": self.center[1]},
            "velocity": {"x": self.velocity[0], "y": self.velocity[1]},
            "state": {
                "visible": self.visible,
                "moving": self.moving,
            },
            "confidence": self.confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------


class WorldState:
    """Scene memory generated from tracked detections.

    Attributes
    ----------
    objects : dict[int, WorldObject]
        Dictionary of objects keyed by tracking id.
    relations : dict[frozenset, Relation]
        Spatial relations between pairs of objects.
    events : deque[str]
        Last *max_events* number of events.
    version : int
        How many updates to the WorldState.
    frame_index : int
        How many frames have been processed.
    """

    def __init__(self, max_events: int = MAX_EVENTS):
        self.critical = False
        self.objects: dict[int, WorldObject] = {}
        self.crops: dict[int, np.ndarray] = {}
        self.frame_width = 0
        self.frame_height = 0
        self.relations: dict[
            set, Relation
        ] = {}  # key: frozenset({id_a, id_b}), value: relation object
        self.events: deque[str] = deque(maxlen=max_events)

        self.version = 0
        self.frame_index = 0
        self._lock = threading.Lock()

    def update_from_detections(self, frame, names, boxes) -> None:
        with self._lock:
            self.critical = False
            self.frame_index += 1
            frame_height, frame_width = frame.shape[:2]
            self.frame_width = int(frame_width)
            self.frame_height = int(frame_height)
            if boxes.id is None:
                return

            self.version += 1
            seen_ids = {int(item) for item in boxes.id.tolist()}
            num_detections = boxes.shape[0]

            for i in range(num_detections):
                track_id = int(boxes.id[i].item())
                xyxy = boxes.xyxy[i]

                crop = safe_crop(frame, xyxy, buffer_px=CROP_BUFFER)
                self.crops[track_id] = crop

                center = (
                    round((boxes.xyxy[i, 0] + boxes.xyxy[i, 2]).item() / 2),
                    round((boxes.xyxy[i, 1] + boxes.xyxy[i, 3]).item() / 2),
                )
                class_name = names[boxes.cls[i].int().item()]
                conf = round(boxes.conf[i].item(), 4)

                if track_id in self.objects:
                    obj = self.objects[track_id]
                    prev_moving_state = obj.moving
                    obj.update(center, conf, self.frame_index, xyxy)

                    if not prev_moving_state and obj.moving:
                        direction = DIRECTION_MAPPING[
                            round(
                                4
                                * math.atan2(-obj.velocity[1], obj.velocity[0])
                                / math.pi
                            )
                        ]
                        self.events.append(
                            f"{obj.type}_{track_id} moved {direction}"
                            + f" at frame {self.frame_index}"
                        )
                    elif prev_moving_state and not obj.moving:
                        self.events.append(
                            f"{obj.type}_{track_id} stopped"
                            + f" at frame {self.frame_index}"
                        )
                else:
                    self.objects[track_id] = WorldObject(
                        track_id,
                        class_name,
                        center,
                        conf,
                        self.frame_index,
                        xyxy,
                    )
                    self.events.append(
                        f"{class_name}_{track_id} appeared"
                        + f" at frame {self.frame_index}"
                    )

            for track_id, obj in self.objects.items():
                is_missing = track_id not in seen_ids
                disappearance_age = self.frame_index - obj.last_seen

                if (
                    is_missing
                    and disappearance_age > DISAPPEARANCE_THRESHOLD
                    and obj.visible
                ):
                    obj.mark_missing()
                    self.critical = True
                    self.events.append(
                        f"{obj.type}_{track_id} disappeared"
                        + f" at frame {self.frame_index}"
                    )

            self._update_relations_delta(seen_ids)

    def _update_relations_delta(self, seen_ids):
        objs = self.objects
        visible_ids = [tid for tid in seen_ids if objs[tid].visible]

        for i in range(len(visible_ids)):
            for j in range(i + 1, len(visible_ids)):
                id_a = visible_ids[i]
                id_b = visible_ids[j]

                a = objs[id_a]
                b = objs[id_b]

                ax, ay = a.center
                bx, by = b.center

                dx = ax - bx
                dy = by - ay  # y-axis flipped

                dist = math.dist(a.center, b.center)

                index = int(round(4 * math.atan2(dy, dx) / math.pi))
                index = max(-4, min(4, index))

                direction = DIRECTION_MAPPING[index]
                overlapping = _bbox_iou(a.xyxy, b.xyxy) > IOU_THRESHOLD

                near = dist < DISTANCE_THRESHOLD

                key = frozenset({id_a, id_b})

                self.relations[key] = Relation(  # type: ignore[index]
                    subject_id=f"{a.type}_{a.track_id}",
                    direction=direction,
                    near=near,
                    overlapping=overlapping,
                    object_id=f"{b.type}_{b.track_id}",
                    last_updated=self.frame_index,
                )

    # Deprecated, useful for getting current graph later
    def _compute_relations(self) -> None:
        self.relations = []  # type: ignore[assignment]

        objs = list(self.objects.values())
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if not a.visible or not b.visible:
                    continue

                ax, ay = a.center
                bx, by = b.center

                dx = ax - bx
                dy = by - ay

                dist = math.dist(a.center, b.center)
                direction = DIRECTION_MAPPING[round(4 * math.atan2(dy, dx) / math.pi)]
                overlap = _bbox_iou(a.xyxy, b.xyxy) > IOU_THRESHOLD

                id_a = f"{a.type}_{a.track_id}"
                id_b = f"{b.type}_{b.track_id}"

                self.relations.append(  # type: ignore[union-attr]
                    Relation(
                        subject_id=id_a,
                        direction=direction,
                        near=dist < DISTANCE_THRESHOLD,
                        overlapping=overlap,
                        object_id=id_b,
                        last_updated=self.frame_index,
                    )
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "critical": self.critical,
                "world_version": self.version,
                "timestamp": self.frame_index,
                "frame_size": {
                    "width": self.frame_width,
                    "height": self.frame_height,
                },
                "objects": [
                    obj.to_dict() for obj in self.objects.values() if obj.visible
                ],
                "relations": [rel.to_dict() for rel in self.relations.values()],
                "recent_events": list(self.events),
            }

    def crop_snapshot(self) -> dict[int, np.ndarray]:
        with self._lock:
            return dict(self.crops)

    def object_types_snapshot(self) -> dict[int, str]:
        with self._lock:
            return {track_id: obj.type for track_id, obj in self.objects.items()}
