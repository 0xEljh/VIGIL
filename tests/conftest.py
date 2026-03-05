from __future__ import annotations

from typing import Any

from vigil.memory.semantic_index import SemanticIndexEntry


class _FakeSemanticIndex:
    def __init__(self, entries: list[SemanticIndexEntry]) -> None:
        self._entries = list(entries)

    def search(self, query: str, top_k: int = 5) -> list[object]:
        del query, top_k
        return []

    def entries_snapshot(self) -> list[SemanticIndexEntry]:
        return list(self._entries)


def _object(
    track_id: int,
    object_type: str = "object",
    *,
    visible: bool = True,
    first_seen: int = 0,
    last_seen: int = 10,
) -> dict[str, object]:
    return {
        "id": track_id,
        "type": object_type,
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 0.0, "y": 0.0},
        "state": {"visible": visible, "moving": False},
        "confidence": 1.0,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def _snapshot(
    world_version: int,
    timestamp: int,
    objects: list[dict[str, object]],
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    return {
        "world_version": world_version,
        "timestamp": timestamp,
        "objects": objects,
        "relations": relations or [],
        "recent_events": [],
    }
