"""Service orchestration tests."""

import pytest

from candidate_ranker.config import Settings
from candidate_ranker.models import JDIntelligence
from candidate_ranker.services import run_pipeline_from_jsonl


@pytest.mark.asyncio
async def test_run_pipeline_from_jsonl_uses_cached_jd_without_ai(monkeypatch, tmp_path) -> None:
    async def fail_extract(*args, **kwargs):
        raise AssertionError("JD extraction should not run when cached intelligence is provided")

    monkeypatch.setattr("candidate_ranker.services.extract_jd_intelligence", fail_extract)
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        (
            '{"candidate_id":"c1","profile":{"current_title":"Backend Engineer",'
            '"years_of_experience":5},"skills":[{"name":"Python"}]}\n'
        ),
        encoding="utf-8",
    )
    schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "profile": {"type": "object"},
            "skills": {"type": "array"},
        },
    }
    settings = Settings(
        openrouter_primary_model="unused",
        top_k_retrieval=10,
        top_k_features=10,
        top_k_explain=10,
        enable_ai_explanations=True,
    )
    jd_intelligence = JDIntelligence(
        title="Backend Engineer",
        required_skills=["Python"],
        preferred_skills=[],
        responsibilities=["Build APIs"],
        experience_years_min=3,
    )

    result = await run_pipeline_from_jsonl(
        candidate_path=str(candidate_path),
        schema=schema,
        settings=settings,
        jd_intelligence=jd_intelligence,
    )

    assert result.jd_intelligence == jd_intelligence
    assert result.ranked[0].candidate_id == "c1"
    assert result.explanations.explanations == []
