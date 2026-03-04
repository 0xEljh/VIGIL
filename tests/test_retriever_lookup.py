from __future__ import annotations

import unittest

from memory.graph_history import GraphHistoryStore
from memory.retriever import MemoryRetriever
from memory.semantic_index import SemanticIndexEntry


class _FakeSemanticIndex:
    def __init__(self, entries: list[SemanticIndexEntry]) -> None:
        self._entries = list(entries)

    def search(self, query: str, top_k: int = 5) -> list[object]:
        del query, top_k
        return []

    def entries_snapshot(self) -> list[SemanticIndexEntry]:
        return list(self._entries)


def _object(
    track_id: int, object_type: str, *, visible: bool = True
) -> dict[str, object]:
    return {
        "id": track_id,
        "type": object_type,
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 0.0, "y": 0.0},
        "state": {"visible": visible, "moving": False},
        "confidence": 1.0,
        "first_seen": 0,
        "last_seen": 10,
    }


def _snapshot(
    world_version: int,
    timestamp: int,
    objects: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "world_version": world_version,
        "timestamp": timestamp,
        "objects": objects,
        "relations": [],
        "recent_events": [],
    }


class MemoryRetrieverLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        semantic_index = _FakeSemanticIndex(
            [
                SemanticIndexEntry(
                    track_id=1,
                    object_type="cup",
                    description="red ceramic cup",
                    indexed_world_version=5,
                ),
                SemanticIndexEntry(
                    track_id=2,
                    object_type="person",
                    description="person standing by the desk",
                    indexed_world_version=6,
                ),
                SemanticIndexEntry(
                    track_id=3,
                    object_type="cup",
                    description="blue plastic cup",
                    indexed_world_version=7,
                ),
            ]
        )

        graph_history = GraphHistoryStore(
            save_interval_world_versions=1,
            max_snapshots=10,
            protected_recent_snapshots=1,
        )
        graph_history.maybe_save(
            _snapshot(
                world_version=5,
                timestamp=100,
                objects=[_object(1, "cup", visible=True)],
            )
        )
        graph_history.maybe_save(
            _snapshot(
                world_version=6,
                timestamp=120,
                objects=[_object(2, "person", visible=True)],
            )
        )

        self.retriever = MemoryRetriever(
            semantic_index=semantic_index,  # type: ignore[arg-type]
            graph_history=graph_history,
        )

    def test_lookup_entities_by_track_id(self) -> None:
        result = self.retriever.lookup_entities(
            track_id=1,
            current_visible_track_ids={1, 2},
        )

        self.assertEqual(result["count"], 1)
        item = result["items"][0]
        self.assertEqual(item["track_id"], 1)
        self.assertEqual(item["object_type"], "cup")
        self.assertTrue(item["currently_visible"])
        self.assertIsNotNone(item["last_visible_graph"])

    def test_lookup_entities_by_class_name(self) -> None:
        result = self.retriever.lookup_entities(
            class_name="cup",
            current_visible_track_ids=set(),
        )

        self.assertEqual(result["count"], 2)
        track_ids = {item["track_id"] for item in result["items"]}
        self.assertEqual(track_ids, {1, 3})

    def test_lookup_entities_without_graph_context(self) -> None:
        result = self.retriever.lookup_entities(
            track_id=1,
            current_visible_track_ids=set(),
            include_graph_context=False,
        )

        self.assertEqual(result["count"], 1)
        item = result["items"][0]
        self.assertIsNone(item["last_visible_graph"])

    def test_lookup_entities_requires_identifier(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.lookup_entities(current_visible_track_ids=set())


if __name__ == "__main__":
    unittest.main()
