from __future__ import annotations

import json
import unittest
from typing import Any

from tests.conftest import _FakeSemanticIndex, _object, _snapshot
from vigil.memory.graph_history import GraphHistoryStore
from vigil.memory.retriever import MemoryRetriever
from vigil.memory.semantic_index import SemanticIndexEntry
from vigil.tools.dispatcher import ToolDispatcher


class ToolDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        semantic_index = _FakeSemanticIndex(
            [
                SemanticIndexEntry(
                    track_id=1,
                    object_type="cup",
                    description="red cup",
                    indexed_world_version=3,
                ),
                SemanticIndexEntry(
                    track_id=2,
                    object_type="cup",
                    description="blue cup",
                    indexed_world_version=3,
                ),
                SemanticIndexEntry(
                    track_id=7,
                    object_type="person",
                    description="person by the table",
                    indexed_world_version=4,
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
                world_version=4,
                timestamp=20,
                objects=[_object(1, "cup", visible=True), _object(7, "person")],
            )
        )

        retriever = MemoryRetriever(
            semantic_index=semantic_index,  # type: ignore[arg-type]
            graph_history=graph_history,
        )

        self._scene_state: dict[str, Any] = {
            "world_version": 9,
            "timestamp": 55,
            "frame_size": {"width": 640, "height": 480},
            "objects": [
                {
                    "id": 1,
                    "type": "cup",
                    "position": {"x": 64, "y": 200},
                    "state": {"visible": True, "moving": False},
                    "confidence": 0.92,
                },
                {
                    "id": 7,
                    "type": "person",
                    "position": {"x": 520, "y": 310},
                    "state": {"visible": True, "moving": False},
                    "confidence": 0.87,
                },
            ],
        }

        self._descriptions = {
            1: "red cup on the left",
            7: "person standing on the right",
        }

        self.dispatcher = ToolDispatcher(
            retriever=retriever,
            scene_state_provider=lambda: dict(self._scene_state),
            text_by_track_id_provider=lambda: dict(self._descriptions),
        )

    def test_lookup_entity_by_track_id(self) -> None:
        result = json.loads(self.dispatcher.dispatch("lookup_entity", {"track_id": 1}))

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["entities"][0]["track_id"], 1)
        self.assertEqual(result["entities"][0]["object_type"], "cup")

    def test_lookup_entity_by_class_name(self) -> None:
        result = json.loads(
            self.dispatcher.dispatch("lookup_entity", {"class_name": "cup"})
        )

        self.assertEqual(result["count"], 2)
        self.assertEqual({item["track_id"] for item in result["entities"]}, {1, 2})

    def test_describe_scene_applies_region_filter(self) -> None:
        result = json.loads(
            self.dispatcher.dispatch(
                "describe_scene",
                {
                    "region": "left",
                    "max_entities": 10,
                },
            )
        )

        self.assertEqual(result["region"], "left")
        self.assertEqual(result["entity_count"], 1)
        self.assertEqual(result["entities"][0]["track_id"], 1)
        self.assertEqual(result["entities"][0]["description"], "red cup on the left")

    def test_unknown_tool_returns_error(self) -> None:
        result = json.loads(self.dispatcher.dispatch("nope", {}))
        self.assertIn("error", result)

    def test_invalid_argument_payload_returns_error(self) -> None:
        result = json.loads(self.dispatcher.dispatch("lookup_entity", "{"))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
