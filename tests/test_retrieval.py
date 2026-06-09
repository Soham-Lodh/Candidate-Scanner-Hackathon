"""Streaming retrieval tests."""

import numpy as np

from candidate_ranker.retrieval import SemanticRetriever


class KeywordRetriever(SemanticRetriever):
    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if "python" in lowered else 0.0])
        return np.asarray(vectors, dtype="float32")


def test_retrieve_stream_keeps_only_top_k_candidates() -> None:
    candidates = (
        {"id": index, "skills": ["Python"] if index in {3, 7} else ["Excel"]}
        for index in range(10)
    )

    retrieved = KeywordRetriever().retrieve_stream("python", candidates, top_k=2, batch_size=3)

    assert len(retrieved) == 2
    assert {candidate["id"] for candidate in retrieved} == {3, 7}
    assert all(candidate["_retrieval_score"] == 1.0 for candidate in retrieved)
