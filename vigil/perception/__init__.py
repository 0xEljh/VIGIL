"""VIGIL perception: world-state tracking with pluggable detection backends.

Core types and data models are always available.  The YOLO detector backend
requires the ``vigil[perception]`` extras group.
"""

from vigil.perception.frame_provider import (
    FramePacket,
    FrameProvider,
    FrameSource,
    FrameSourceMode,
    FrameSourceSelector,
    SourceDecision,
    frame_provider,
)
from vigil.perception.types import DetectionProvider, DetectionResult
from vigil.perception.world_state import (
    Relation,
    WorldObject,
    WorldState,
    safe_crop,
)

__all__ = [
    "DetectionProvider",
    "DetectionResult",
    "FramePacket",
    "FrameProvider",
    "FrameSource",
    "FrameSourceMode",
    "FrameSourceSelector",
    "Relation",
    "SourceDecision",
    "WorldObject",
    "WorldState",
    "frame_provider",
    "safe_crop",
]
