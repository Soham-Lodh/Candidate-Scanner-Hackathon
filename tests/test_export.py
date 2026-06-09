"""Export tests for recruiter-facing ranking output."""

import csv
from io import StringIO

from candidate_ranker.export import TOP_CANDIDATE_EXPORT_LIMIT, ranking_export_rows, rankings_to_csv
from candidate_ranker.models import CandidateExplanation, CandidateScore


def test_ranking_export_rows_are_limited_and_use_requested_columns() -> None:
    scores = [_score(str(index)) for index in range(TOP_CANDIDATE_EXPORT_LIMIT + 5)]

    rows = ranking_export_rows(scores)

    assert len(rows) == TOP_CANDIDATE_EXPORT_LIMIT
    assert list(rows[0]) == ["candidate_id", "rank", "score", "reasoning"]
    assert rows[0]["rank"] == 1
    assert rows[-1]["rank"] == TOP_CANDIDATE_EXPORT_LIMIT


def test_rankings_to_csv_uses_scoring_reasoning_instead_of_generic_ai_text() -> None:
    scores = [_score("abc")]
    explanations = [
        CandidateExplanation(
            candidate_id="abc",
            summary="Strong candidate",
            recruiter_rationale=["Built production Python systems", "Matches data requirements"],
            interview_focus=[],
            risk_notes=[],
        )
    ]

    csv_data = rankings_to_csv(scores, explanations)
    rows = list(csv.DictReader(StringIO(csv_data)))

    assert rows[0]["candidate_id"] == "abc"
    assert rows[0]["rank"] == "1"
    assert rows[0]["score"] == "91.5"
    assert "Candidate demonstrates relevant Python capability" in rows[0]["reasoning"]
    assert "Key strengths:" in rows[0]["reasoning"]
    assert "Potential concerns:" in rows[0]["reasoning"]


def _score(candidate_id: str) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        display_name=f"Candidate {candidate_id}",
        composite_score=91.5,
        breakdown={"skill_overlap": 95.0},
        matched_skills=["Python"],
        missing_skills=[],
        strengths=["Strong Python experience"],
        concerns=[],
        integrity_flags=[],
        raw_candidate={},
    )
