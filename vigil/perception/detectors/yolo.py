"""YOLO-based detection and tracking loop.

This module contains the YOLO+BoT-SORT tracking loop extracted from the
original monolithic ``yoloworldstate.py``.  It depends on ``ultralytics``
and ``opencv-python`` and is installed as part of the ``vigil[perception]``
extras group.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Callable, cast

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]

from vigil.perception.frame_provider import (
    FrameSourceMode,
    FrameSourceSelector,
    frame_provider,
)
from vigil.perception.world_state import WorldState

IDLE_SLEEP_SECONDS = 0.01
WAIT_LOG_INTERVAL_SECONDS = 2.0


def run_world_state_tracking_loop(
    state: WorldState | None = None,
    camera_index: int = 0,
    model_path: str = "yolo26x.pt",
    tracker_path: str = "config/botsort.yaml",
    display: bool = True,
    on_snapshot: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    frame_source_mode: FrameSourceMode = "auto",
    api_stale_after_seconds: float = 1.0,
    switch_to_api_after_consecutive: int = 5,
    switch_cooldown_seconds: float = 2.0,
) -> WorldState:
    world_state = state or WorldState()

    selector = FrameSourceSelector(
        mode=frame_source_mode,
        api_stale_after_seconds=api_stale_after_seconds,
        switch_to_api_after_consecutive=switch_to_api_after_consecutive,
        switch_cooldown_seconds=switch_cooldown_seconds,
    )

    local_cap: cv2.VideoCapture | None = None
    if frame_source_mode in {"auto", "local"}:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            local_cap = cap
            if display:
                frame_width = int(local_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(frame_width, frame_height)
        elif frame_source_mode == "local":
            cap.release()
            raise RuntimeError(
                "Cannot open video device. Check camera index or connection."
            )
        else:
            cap.release()
            print(
                "Local camera unavailable. Waiting for API stream frames in auto mode."
            )

    if frame_source_mode == "api":
        print("Frame source mode: api (strict)")
    elif frame_source_mode == "local":
        print("Frame source mode: local (strict)")
    else:
        print("Frame source mode: auto (prefer API, fallback local)")

    model = YOLO(model_path)
    last_wait_log_at = 0.0
    display_enabled = display
    display_error_reported = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            now = time.time()
            packet = frame_provider.get_frame()
            decision = selector.select_source(
                packet,
                local_available=local_cap is not None,
                now=now,
            )

            if decision.switched:
                reason = decision.reason or "unknown"
                print(f"Frame source switched to {decision.source} ({reason})")

            frame: np.ndarray | None = None
            wait_reason: str | None = None
            if decision.source == "api":
                if selector.is_api_packet_fresh(packet, now=now):
                    frame = packet.frame
                else:
                    wait_reason = "api"
            else:
                if local_cap is None:
                    wait_reason = "local"
                else:
                    has_frame, local_frame = local_cap.read()
                    if has_frame:
                        frame = local_frame
                    else:
                        wait_reason = "local"

            if frame is None:
                if (now - last_wait_log_at) >= WAIT_LOG_INTERVAL_SECONDS:
                    if wait_reason == "api":
                        frame_age = frame_provider.frame_age_seconds(now=now)
                        if frame_age is None:
                            print("Waiting for API frames...")
                        else:
                            print(
                                f"Waiting for fresh API frame (latest age: {frame_age:.2f}s)"
                            )
                    else:
                        print("Waiting for local camera frame...")
                    last_wait_log_at = now

                if display_enabled:
                    try:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        if not display_error_reported:
                            print(
                                "OpenCV display backend unavailable. Disabling display; "
                                "use --no-display to suppress this warning."
                            )
                            display_error_reported = True
                        display_enabled = False

                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                verbose=False,
            )
            if not results:
                continue

            world_state.update_from_detections(
                frame, results[0].names, results[0].boxes
            )

            if on_snapshot is not None:
                on_snapshot(world_state.snapshot())

            if display_enabled:
                try:
                    annotated_frame = results[0].plot()
                    cv2.imshow("YOLO26 Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    if not display_error_reported:
                        print(
                            "OpenCV display backend unavailable. Disabling display; "
                            "use --no-display to suppress this warning."
                        )
                        display_error_reported = True
                    display_enabled = False
    finally:
        if local_cap is not None:
            local_cap.release()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return world_state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO world-state tracking with configurable frame source mode."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index"
    )
    parser.add_argument("--model-path", default="yolo26x.pt", help="YOLO model path")
    parser.add_argument(
        "--tracker-path",
        default="config/botsort.yaml",
        help="BoT-SORT tracker config path",
    )
    parser.add_argument(
        "--frame-source-mode",
        "--frame-source",
        choices=["auto", "api", "local"],
        default="auto",
        help="Frame source mode: auto prefers API, api/local are strict.",
    )
    parser.add_argument(
        "--api-stale-after-seconds",
        type=float,
        default=1.0,
        help="How old API frames can be before treated as stale.",
    )
    parser.add_argument(
        "--switch-to-api-after-consecutive",
        type=int,
        default=5,
        help="Fresh API streak required before switching local -> API.",
    )
    parser.add_argument(
        "--switch-cooldown-seconds",
        type=float,
        default=2.0,
        help="Minimum time between source switches in auto mode.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="output json to be sent to LLM for debugging",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.frame_source_mode == "api":
        print(
            "[api] Standalone perception CLI does not start embedded ingestion. "
            "Use 'python -m examples.embodied_agent.orchestration.pipeline' "
            "for embedded API ingest."
        )

    final_state = run_world_state_tracking_loop(
        camera_index=args.camera_index,
        model_path=args.model_path,
        tracker_path=args.tracker_path,
        display=not args.no_display,
        frame_source_mode=cast(FrameSourceMode, args.frame_source_mode),
        api_stale_after_seconds=args.api_stale_after_seconds,
        switch_to_api_after_consecutive=args.switch_to_api_after_consecutive,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
    )

    print(final_state.snapshot())
    if args.output_json:
        with open("output.json", "w") as json_file:
            json.dump(final_state.snapshot(), json_file, indent=4)


if __name__ == "__main__":
    main()
