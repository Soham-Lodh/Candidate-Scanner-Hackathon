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


def test_ranking_export_rows_break_score_ties_by_candidate_id() -> None:
    rows = ranking_export_rows([_score("CAND_0000002"), _score("CAND_0000001")])

    assert rows[0]["candidate_id"] == "CAND_0000001"
    assert rows[1]["candidate_id"] == "CAND_0000002"


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
    assert rows[0]["score"] == "1.00000000"
    assert "Candidate demonstrates relevant Python capability" in rows[0]["reasoning"]
    assert "Key strengths:" in rows[0]["reasoning"]
    assert "Potential concerns:" in rows[0]["reasoning"]


def test_ranking_export_rows_normalize_only_exported_scores() -> None:
    scores = [
        _score("CAND_0000001", 52.11),
        _score("CAND_0000002", 51.67),
        _score("CAND_0000003", 51.11),
    ]
    original_ids = [score.candidate_id for score in scores]
    original_scores = [score.composite_score for score in scores]
    original_reasoning = [ranking_export_rows([score])[0]["reasoning"] for score in scores]

    rows = ranking_export_rows(scores)

    assert [row["candidate_id"] for row in rows] == original_ids
    assert [score.composite_score for score in scores] == original_scores
    assert [row["reasoning"] for row in rows] == original_reasoning
    assert [row["score"] for row in rows] == ["1.00000000", "0.60400000", "0.10000000"]


def test_ranking_export_rows_normalize_equal_scores_to_one() -> None:
    rows = ranking_export_rows([_score("CAND_0000001", 51.11), _score("CAND_0000002", 51.11)])

    assert [row["score"] for row in rows] == ["1.00000000", "1.00000000"]


def test_scoring_reasoning_uses_candidate_specific_redrob_evidence() -> None:
    score = CandidateScore(
        candidate_id="CAND_0000001",
        display_name="Candidate 1",
        composite_score=88.0,
        breakdown={},
        matched_skills=["Python", "NLP"],
        missing_skills=["Vector databases"],
        strengths=[],
        concerns=[],
        integrity_flags=[],
        raw_candidate={
            "profile": {
                "current_title": "Backend Engineer",
                "years_of_experience": 6.9,
                "location": "Toronto",
                "country": "Canada",
            },
            "skills": [
                {
                    "name": "NLP",
                    "proficiency": "advanced",
                    "endorsements": 37,
                    "duration_months": 26,
                },
                {
                    "name": "React",
                    "proficiency": "intermediate",
                    "endorsements": 6,
                    "duration_months": 35,
                },
            ],
            "redrob_signals": {
                "recruiter_response_rate": 0.34,
                "notice_period_days": 60,
                "preferred_work_mode": "onsite",
                "skill_assessment_scores": {"NLP": 50},
            },
        },
    )

    reasoning = rankings_to_csv([score])
    rows = list(csv.DictReader(StringIO(reasoning)))

    assert "Backend Engineer with 6.9 yrs" in rows[0]["reasoning"]
    assert "1 matched JD skills: NLP (advanced, 26 mo, 37 endorsements)" in rows[0]["reasoning"]
    assert "response rate 0.34" in rows[0]["reasoning"]
    assert "gaps: Vector databases" in rows[0]["reasoning"]


def _score(candidate_id: str, composite_score: float = 91.5) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        display_name=f"Candidate {candidate_id}",
        composite_score=composite_score,
        breakdown={"skill_overlap": 95.0},
        matched_skills=["Python"],
        missing_skills=[],
        strengths=["Strong Python experience"],
        concerns=[],
        integrity_flags=[],
        raw_candidate={},
    )
