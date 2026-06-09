"""Semantic retrieval funnel using sentence-transformers and FAISS with safe fallback."""

from __future__ import annotations

import hashlib
import heapq
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from candidate_ranker.text import normalize_text

LOGGER = logging.getLogger(__name__)


class SemanticRetriever:
    """Retrieve likely matches before expensive local feature scoring."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def retrieve(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Return the top candidate records by semantic similarity."""

        if not candidates:
            return []
        texts = [normalize_text(candidate) for candidate in candidates]
        vectors = self._embed(texts)
        query_vector = self._embed([query])
        scores = self._search(vectors, query_vector[0])
        order = np.argsort(-scores)[: min(top_k, len(candidates))]
        LOGGER.info("Retrieved %s/%s candidates", len(order), len(candidates))
        return [dict(candidates[int(index)], _retrieval_score=float(scores[int(index)])) for index in order]

    def retrieve_stream(
        self,
        query: str,
        candidates: Iterable[dict[str, Any]],
        top_k: int,
        batch_size: int = 256,
    ) -> list[dict[str, Any]]:
        """Return top candidates from an iterable without materializing all records."""

        if top_k <= 0:
            return []
        query_vector = self._embed([query])[0]
        heap: list[tuple[float, int, dict[str, Any]]] = []
        batch: list[dict[str, Any]] = []
        seen = 0
        for candidate in candidates:
            batch.append(candidate)
            if len(batch) >= batch_size:
                seen += self._score_batch(batch, query_vector, heap, top_k, seen)
                batch = []
        if batch:
            seen += self._score_batch(batch, query_vector, heap, top_k, seen)
        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        LOGGER.info("Stream-retrieved %s/%s candidates", len(ranked), seen)
        return [dict(candidate, _retrieval_score=float(score)) for score, _, candidate in ranked]

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32")
        except Exception as exc:
            LOGGER.warning("Embedding model unavailable; using deterministic fallback: %s", exc)
            return np.vstack([_hash_embedding(text) for text in texts]).astype("float32")

    def _score_batch(
        self,
        batch: list[dict[str, Any]],
        query_vector: np.ndarray,
        heap: list[tuple[float, int, dict[str, Any]]],
        top_k: int,
        seen_before: int,
    ) -> int:
        texts = [normalize_text(candidate) for candidate in batch]
        vectors = self._embed(texts)
        scores = vectors @ query_vector
        for offset, score in enumerate(scores):
            sequence = seen_before + offset
            item = (float(score), sequence, batch[offset])
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
        return len(batch)

    @staticmethod
    def _search(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            scores, _ = index.search(query_vector.reshape(1, -1), len(vectors))
            flat = np.zeros(len(vectors), dtype="float32")
            ranked = scores[0]
            ids = index.search(query_vector.reshape(1, -1), len(vectors))[1][0]
            for position, candidate_index in enumerate(ids):
                flat[candidate_index] = ranked[position]
            return flat
        except Exception as exc:
            LOGGER.warning("FAISS unavailable; using NumPy dot search: %s", exc)
            return vectors @ query_vector


def _hash_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    vector = np.zeros(dimensions, dtype="float32")
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
