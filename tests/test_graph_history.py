from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


_GRAPH_HISTORY_PATH = (
    Path(__file__).resolve().parents[1] / "memory" / "graph_history.py"
)
_GRAPH_HISTORY_SPEC = spec_from_file_location(
    "graph_history_module",
    _GRAPH_HISTORY_PATH,
)
if _GRAPH_HISTORY_SPEC is None or _GRAPH_HISTORY_SPEC.loader is None:
    raise ImportError(f"Failed to load graph history module from {_GRAPH_HISTORY_PATH}")

_GRAPH_HISTORY_MODULE = module_from_spec(_GRAPH_HISTORY_SPEC)
sys.modules[_GRAPH_HISTORY_SPEC.name] = _GRAPH_HISTORY_MODULE
_GRAPH_HISTORY_SPEC.loader.exec_module(_GRAPH_HISTORY_MODULE)

GraphHistoryStore = _GRAPH_HISTORY_MODULE.GraphHistoryStore


def _object(
    track_id: int,
    *,
    visible: bool = True,
    first_seen: int = 0,
    last_seen: int = 0,
    object_type: str = "object",
) -> dict[str, object]:
    return {
        "id": track_id,
        "type": object_type,
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 0.0, "y": 0.0},
        "state": {
            "visible": visible,
            "moving": False,
        },
        "confidence": 1.0,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def _relation(subject: str, object_id: str) -> dict[str, object]:
    return {
        "subject": subject,
        "relation": "left, near, not overlapping",
        "object": object_id,
        "last_updated": 0,
    }


def _snapshot(
    world_version: int,
    timestamp: int,
    objects: list[dict[str, object]],
    relations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "world_version": world_version,
        "timestamp": timestamp,
        "objects": objects,
        "relations": relations or [],
        "recent_events": [],
    }


class GraphHistoryStoreTests(unittest.TestCase):
    def test_prunes_to_visible_and_first_hop_related_entities(self) -> None:
        store = GraphHistoryStore(
            save_interval_world_versions=1,
            max_snapshots=10,
            protected_recent_snapshots=1,
        )

        snapshot = _snapshot(
            world_version=1,
            timestamp=100,
            objects=[
                _object(1, visible=True, object_type="person"),
                _object(2, visible=False, object_type="cup"),
                _object(3, visible=False, object_type="bottle"),
            ],
            relations=[
                _relation("person_1", "cup_2"),
                _relation("cup_2", "bottle_3"),
            ],
        )

        saved = store.maybe_save(snapshot)
        self.assertTrue(saved)

        history = store.snapshots()
        self.assertEqual(len(history), 1)

        stored = history[0]
        self.assertEqual(stored.visible_track_ids, frozenset({1}))

        object_ids = {
            int(item["id"])
            for item in stored.graph["objects"]
            if isinstance(item, dict) and "id" in item
        }
        self.assertEqual(object_ids, {1, 2})

        relations = stored.graph["relations"]
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["subject"], "person_1")
        self.assertEqual(relations[0]["object"], "cup_2")

    def test_time_in_frame_rms_uses_only_visible_entities(self) -> None:
        store = GraphHistoryStore(
            save_interval_world_versions=1,
            max_snapshots=10,
            protected_recent_snapshots=1,
        )

        snapshot = _snapshot(
            world_version=1,
            timestamp=20,
            objects=[
                _object(1, visible=True, first_seen=5, last_seen=15),
                _object(2, visible=False, first_seen=0, last_seen=100),
            ],
        )

        self.assertTrue(store.maybe_save(snapshot))
        stored = store.snapshots()[0]
        self.assertEqual(stored.visible_entity_time_in_frame_rms, 10.0)

    def test_description_rms_is_computed_once_per_snapshot(self) -> None:
        store = GraphHistoryStore(
            save_interval_world_versions=1,
            max_snapshots=10,
            protected_recent_snapshots=1,
        )

        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=1,
                    timestamp=10,
                    objects=[_object(1, visible=True, first_seen=0, last_seen=10)],
                ),
                text_by_track_id={1: "x" * 20},
            )
        )
        first_saved = store.snapshots()[0]
        self.assertEqual(first_saved.description_length_rms, 20.0)

        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=2,
                    timestamp=11,
                    objects=[_object(2, visible=True, first_seen=0, last_seen=11)],
                ),
                text_by_track_id={2: "short"},
            )
        )

        history = store.snapshots()
        self.assertEqual(history[0].description_length_rms, 20.0)

    def test_protects_most_recent_snapshots_from_eviction(self) -> None:
        store = GraphHistoryStore(
            save_interval_world_versions=1,
            max_snapshots=3,
            protected_recent_snapshots=1,
            random_seed=0,
        )

        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=1,
                    timestamp=100,
                    objects=[_object(1, visible=True, first_seen=0, last_seen=1)],
                )
            )
        )
        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=2,
                    timestamp=200,
                    objects=[_object(2, visible=True, first_seen=0, last_seen=2)],
                )
            )
        )
        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=3,
                    timestamp=300,
                    objects=[_object(3, visible=True, first_seen=0, last_seen=3)],
                )
            )
        )

        self.assertTrue(
            store.maybe_save(
                _snapshot(
                    world_version=4,
                    timestamp=400,
                    objects=[_object(4, visible=True, first_seen=0, last_seen=4)],
                )
            )
        )

        versions = {item.world_version for item in store.snapshots()}
        self.assertIn(3, versions)
        self.assertIn(4, versions)
        self.assertEqual(len(versions), 3)

    def test_probabilistic_eviction_prefers_lower_scored_snapshots(self) -> None:
        low_evicted = 0
        high_evicted = 0

        for seed in range(200):
            store = GraphHistoryStore(
                save_interval_world_versions=1,
                max_snapshots=2,
                protected_recent_snapshots=0,
                random_seed=seed,
            )

            self.assertTrue(
                store.maybe_save(
                    _snapshot(
                        world_version=1,
                        timestamp=1,
                        objects=[
                            _object(1, visible=True, first_seen=0, last_seen=5),
                        ],
                    ),
                    text_by_track_id={1: "small"},
                )
            )

            self.assertTrue(
                store.maybe_save(
                    _snapshot(
                        world_version=2,
                        timestamp=2,
                        objects=[
                            _object(2, visible=True, first_seen=0, last_seen=1000),
                            _object(20, visible=True, first_seen=0, last_seen=900),
                            _object(21, visible=True, first_seen=0, last_seen=800),
                        ],
                    ),
                    text_by_track_id={
                        2: "x" * 120,
                        20: "x" * 140,
                        21: "x" * 160,
                    },
                )
            )

            self.assertTrue(
                store.maybe_save(
                    _snapshot(
                        world_version=3,
                        timestamp=3,
                        objects=[_object(3, visible=True, first_seen=0, last_seen=3)],
                    )
                )
            )

            remaining_versions = {item.world_version for item in store.snapshots()}
            if 1 not in remaining_versions:
                low_evicted += 1
            if 2 not in remaining_versions:
                high_evicted += 1

        self.assertGreater(low_evicted, high_evicted)


if __name__ == "__main__":
    unittest.main()
