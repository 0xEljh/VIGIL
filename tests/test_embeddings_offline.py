from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from vigil.memory.embeddings import EmbeddingModel


class _FakeSentenceTransformer:
    init_calls: list[tuple[str, bool]] = []

    def __init__(self, model_name_or_path: str, *, local_files_only: bool = False):
        self._dimension = 3
        self._local_files_only = local_files_only
        self._model_name_or_path = model_name_or_path
        type(self).init_calls.append((model_name_or_path, local_files_only))

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        del normalize_embeddings, convert_to_numpy
        return np.ones((len(texts), self._dimension), dtype=np.float32)


class EmbeddingModelOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_module = sys.modules.get("sentence_transformers")
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = _FakeSentenceTransformer
        sys.modules["sentence_transformers"] = module
        _FakeSentenceTransformer.init_calls.clear()

    def tearDown(self) -> None:
        if self._original_module is None:
            sys.modules.pop("sentence_transformers", None)
            return
        sys.modules["sentence_transformers"] = self._original_module

    def test_offline_model_uses_local_files_only(self) -> None:
        model = EmbeddingModel(offline=True)

        vector = model.encode("hello")

        self.assertEqual(vector.shape, (3,))
        self.assertEqual(
            _FakeSentenceTransformer.init_calls,
            [(model.model_name, True)],
        )

    def test_online_model_does_not_force_local_files_only(self) -> None:
        model = EmbeddingModel(offline=False)

        model.encode("hello")

        self.assertEqual(
            _FakeSentenceTransformer.init_calls,
            [(model.model_name, False)],
        )


if __name__ == "__main__":
    unittest.main()
