"""Candidate ingestion tests."""

from io import BytesIO

from candidate_ranker.ingestion import read_candidates


def test_read_candidates_accepts_file_like_jsonl() -> None:
    uploaded_file = BytesIO(b'{"id":"c1","skills":["Python"]}\n{"id":"c2","skills":["SQL"]}\n')

    candidates = read_candidates("candidates.jsonl", uploaded_file)

    assert candidates == [
        {"id": "c1", "skills": ["Python"], "_row_id": 1},
        {"id": "c2", "skills": ["SQL"], "_row_id": 2},
    ]
