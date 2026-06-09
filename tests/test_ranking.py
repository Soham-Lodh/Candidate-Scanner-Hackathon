"""Ranking tests for local feature scoring."""

from candidate_ranker.models import JDIntelligence
from candidate_ranker.ranking import rank_candidates, rank_candidates_stream
from candidate_ranker.schema_mapping import build_schema_map


def test_rank_candidates_scores_skill_match() -> None:
    schema = {
        "type": "object",
        "properties": {
            "person": {"type": "object", "properties": {"name": {"type": "string"}}},
            "toolkit": {"type": "array", "items": {"type": "string"}},
            "background_years": {"type": "number"},
        },
    }
    schema_map = build_schema_map(schema)
    jd = JDIntelligence(required_skills=["Python", "SQL"], preferred_skills=["Streamlit"])
    candidates = [
        {"person": {"name": "A"}, "toolkit": ["Python", "SQL", "Streamlit"], "background_years": 6},
        {"person": {"name": "B"}, "toolkit": ["Excel"], "background_years": 1},
    ]
    ranked = rank_candidates(candidates, jd, schema_map, top_k=2)
    assert ranked[0].display_name == "A"
    assert ranked[0].composite_score > ranked[1].composite_score


def test_rank_candidates_uses_candidate_id_and_profile_name_separately() -> None:
    schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "profile": {
                "type": "object",
                "properties": {
                    "anonymized_name": {"type": "string"},
                    "years_of_experience": {"type": "number"},
                },
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }
    schema_map = build_schema_map(schema)
    jd = JDIntelligence(required_skills=["Python"])
    candidates = [
        {
            "candidate_id": "CAND_001",
            "profile": {"anonymized_name": "Rohan Mukherjee", "years_of_experience": 5},
            "skills": [{"name": "Python"}],
        }
    ]

    ranked = rank_candidates(candidates, jd, schema_map, top_k=1)

    assert ranked[0].candidate_id == "CAND_001"
    assert ranked[0].display_name == "Rohan Mukherjee"


def test_rank_candidates_stream_keeps_best_scores_without_materializing_all() -> None:
    schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "profile": {"type": "object", "properties": {"anonymized_name": {"type": "string"}}},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
    }
    schema_map = build_schema_map(schema)
    jd = JDIntelligence(required_skills=["Python"], preferred_skills=["FastAPI"])
    candidates = (
        {
            "candidate_id": f"CAND_{index}",
            "profile": {"anonymized_name": f"Candidate {index}"},
            "skills": ["Python", "FastAPI"] if index == 42 else ["Excel"],
        }
        for index in range(100)
    )

    ranked = rank_candidates_stream(candidates, jd, schema_map, top_k=5)

    assert len(ranked) == 5
    assert ranked[0].candidate_id == "CAND_42"
