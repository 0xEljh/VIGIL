from __future__ import annotations

from vigil.memory import (
    EmbeddingModel,
    GraphHistoryStore,
    MemoryRetriever,
    SemanticIndex,
)


def main() -> None:
    embedding_model = EmbeddingModel()
    semantic_index = SemanticIndex(embedding_model)
    graph_history = GraphHistoryStore(save_interval_world_versions=1, max_snapshots=10)
    retriever = MemoryRetriever(
        semantic_index=semantic_index, graph_history=graph_history
    )

    snapshot = {
        "world_version": 1,
        "timestamp": 1,
        "objects": [
            {
                "id": 1,
                "type": "cup",
                "position": {"x": 10, "y": 20},
                "velocity": {"x": 0, "y": 0},
                "state": {"visible": True, "moving": False},
                "confidence": 0.99,
                "first_seen": 1,
                "last_seen": 1,
            }
        ],
        "relations": [],
        "recent_events": ["cup_1 appeared at frame 1"],
    }
    graph_history.maybe_save(snapshot, text_by_track_id={1: "red ceramic mug"})

    semantic_index.add(
        track_id=1,
        object_type="cup",
        description="red ceramic mug on a desk",
        indexed_world_version=1,
    )

    result = retriever.query_context(
        "where is the mug",
        top_k=1,
        current_visible_track_ids={1},
    )
    print(result)


if __name__ == "__main__":
    main()
