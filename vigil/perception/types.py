"""Abstraction layer for pluggable detection backends.

The :class:`DetectionProvider` protocol defines the minimal interface that any
object-detection backend must satisfy in order to drive
:class:`~vigil.perception.world_state.WorldState`.  The bundled YOLO
implementation (``vigil.perception.detectors.yolo``) is the default provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """A single frame's worth of detection output.

    Attributes
    ----------
    names : dict[int, str]
        Mapping from class index to human-readable class name.
    boxes_xyxy : np.ndarray
        ``(N, 4)`` array of bounding boxes in ``[x1, y1, x2, y2]`` format.
    confidences : np.ndarray
        ``(N,)`` array of detection confidence scores.
    class_indices : np.ndarray
        ``(N,)`` integer array of class indices into *names*.
    track_ids : np.ndarray | None
        ``(N,)`` integer array of tracker-assigned IDs, or ``None`` when the
        backend does not provide tracking.
    """

    names: dict[int, str]
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_indices: np.ndarray
    track_ids: np.ndarray | None = field(default=None)


@runtime_checkable
class DetectionProvider(Protocol):
    """Protocol that every detection backend must implement.

    A provider wraps a detection (and optionally tracking) model.  The single
    required method, :meth:`detect`, receives a BGR frame and returns a
    :class:`DetectionResult`.
    """

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection (and optional tracking) on *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as a NumPy array with shape ``(H, W, 3)``.

        Returns
        -------
        DetectionResult
            Detections for the frame.
        """
        ...
