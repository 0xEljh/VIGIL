from __future__ import annotations

import unittest
from typing import Any

from examples.embodied_agent.orchestration.aux_context import (
    _build_auxiliary_context_with_fallback,
)


class _FailingSemanticRetriever:
    def __init__(self) -> None:
        self.all_entity_calls: list[set[int]] = []
        self.query_calls: list[tuple[str, int, set[int]]] = []

    def all_entity_memory_context(
        self, *, current_visible_track_ids: set[int]
    ) -> dict[str, Any]:
        self.all_entity_calls.append(set(current_visible_track_ids))
        return {
            "count": 1,
            "items": [
                {
                    "track_id": 1,
                    "object_type": "cup",
                    "description": "blue mug",
                    "currently_visible": 1 in current_visible_track_ids,
                }
            ],
        }

    def query_context(
        self,
        query: str,
        *,
        top_k: int,
        current_visible_track_ids: set[int],
    ) -> dict[str, Any]:
        self.query_calls.append((query, top_k, set(current_visible_track_ids)))
        raise RuntimeError("semantic search unavailable")


class PipelineAuxiliaryContextTests(unittest.TestCase):
    def test_falls_back_when_index_and_semantic_search_fail(self) -> None:
        retriever = _FailingSemanticRetriever()
        warnings: list[str] = []

        def _fail_index() -> None:
            raise RuntimeError("embedding model unavailable")

        context = _build_auxiliary_context_with_fallback(
            scene_state={
                "world_version": 10,
                "objects": [
                    {"id": 1},
                    {"id": "2"},
                    {"id": None},
                ],
            },
            query_text="where is the mug",
            descriptions_snapshot={"items": []},
            memory_retriever=retriever,
            semantic_search_top_k=5,
            index_descriptions=_fail_index,
            warning_state={},
            warning_reporter=warnings.append,
        )

        self.assertIn("object_descriptions", context)
        self.assertIn("entity_memory", context)
        self.assertIn("semantic_entity_matches", context)
        self.assertEqual(
            context["semantic_entity_matches"],
            {
                "query": "where is the mug",
                "count": 0,
                "items": [],
            },
        )
        self.assertEqual(retriever.all_entity_calls, [{1, 2}])
        self.assertEqual(retriever.query_calls, [("where is the mug", 5, {1, 2})])
        self.assertEqual(len(warnings), 2)

    def test_warns_once_per_failure_kind_with_shared_warning_state(self) -> None:
        retriever = _FailingSemanticRetriever()
        warnings: list[str] = []
        warning_state: dict[str, bool] = {}

        def _fail_index() -> None:
            raise RuntimeError("index fail")

        _build_auxiliary_context_with_fallback(
            scene_state={"objects": [{"id": 3}]},
            query_text="find bottle",
            descriptions_snapshot={"items": []},
            memory_retriever=retriever,
            semantic_search_top_k=3,
            index_descriptions=_fail_index,
            warning_state=warning_state,
            warning_reporter=warnings.append,
        )

        _build_auxiliary_context_with_fallback(
            scene_state={"objects": [{"id": 3}]},
            query_text="find bottle",
            descriptions_snapshot={"items": []},
            memory_retriever=retriever,
            semantic_search_top_k=3,
            index_descriptions=_fail_index,
            warning_state=warning_state,
            warning_reporter=warnings.append,
        )

        self.assertEqual(len(warnings), 2)
        self.assertIn("Semantic indexing unavailable", warnings[0])
        self.assertIn("Semantic search unavailable", warnings[1])


if __name__ == "__main__":
    unittest.main()
