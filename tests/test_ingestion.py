"""Candidate ingestion tests."""

import gzip
from io import BytesIO

from candidate_ranker.ingestion import iter_jsonl_candidates, read_candidates


def test_read_candidates_accepts_file_like_jsonl() -> None:
    uploaded_file = BytesIO(b'{"id":"c1","skills":["Python"]}\n{"id":"c2","skills":["SQL"]}\n')

    candidates = read_candidates("candidates.jsonl", uploaded_file)

    assert candidates == [
        {"id": "c1", "skills": ["Python"], "_row_id": 1},
        {"id": "c2", "skills": ["SQL"], "_row_id": 2},
    ]


def test_iter_jsonl_candidates_accepts_gzip_jsonl(tmp_path) -> None:
    path = tmp_path / "candidates.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b'{"id":"c1","skills":["Python"]}\n')

    assert list(iter_jsonl_candidates(path)) == [
        {"id": "c1", "skills": ["Python"], "_row_id": 1},
    ]
