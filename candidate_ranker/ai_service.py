"""Typed AI workflows for the two permitted model calls."""

from __future__ import annotations

import json
import logging

from ai.openrouter_client import OpenRouterClient
from candidate_ranker.export import scoring_reasoning
from candidate_ranker.models import CandidateScore, ExplanationBatch, JDIntelligence, SchemaMap

LOGGER = logging.getLogger(__name__)


async def extract_jd_intelligence(
    jd_text: str,
    schema_map: SchemaMap,
    *,
    model: str,
    client: OpenRouterClient | None = None,
) -> JDIntelligence:
    """AI call #1: extract JD and schema intelligence as strict JSON."""

    openrouter = client or OpenRouterClient()
    messages = [
        {
            "role": "system",
            "content": (
                "You extract recruiting intelligence. Return only strict JSON matching this shape: "
                '{"title": str, "seniority": str, "required_skills": [str], "preferred_skills": [str], '
                '"responsibilities": [str], "locations": [str], "education_keywords": [str], '
                '"experience_years_min": number|null, "production_signals": [str], '
                '"behavioral_signals": [str], "schema_notes": [str]}.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"job_description": jd_text[:24000], "schema_map": schema_map.model_dump()},
                ensure_ascii=True,
            ),
        },
    ]
    result = await openrouter.chat_json(
        messages,
        JDIntelligence,
        model=model,
        temperature=0.0,
        max_tokens=1800,
        deterministic=True,
    )
    return result if isinstance(result, JDIntelligence) else JDIntelligence.model_validate(result)


async def generate_candidate_explanations(
    jd: JDIntelligence,
    candidates: list[CandidateScore],
    *,
    model: str,
    client: OpenRouterClient | None = None,
) -> ExplanationBatch:
    """AI call #2: generate explanations for top candidates as strict JSON."""

    openrouter = client or OpenRouterClient()
    compact_candidates = []
    for candidate in candidates:
        payload = candidate.model_dump(
            include={
                "candidate_id",
                "display_name",
                "composite_score",
                "breakdown",
                "matched_skills",
                "missing_skills",
                "strengths",
                "concerns",
                "integrity_flags",
            }
        )
        payload["scoring_reasoning"] = scoring_reasoning(candidate)
        compact_candidates.append(payload)
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise recruiter explanations grounded only in the provided scoring_reasoning, "
                "breakdown, matched skills, strengths, and concerns. Return only strict JSON matching: "
                '{"explanations":[{"candidate_id":str,"summary":str,"recruiter_rationale":[str],'
                '"interview_focus":[str],"risk_notes":[str]}]}. Do not invent facts.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"jd_intelligence": jd.model_dump(), "ranked_candidates": compact_candidates},
                ensure_ascii=True,
            ),
        },
    ]
    result = await openrouter.chat_json(
        messages,
        ExplanationBatch,
        model=model,
        temperature=0.2,
        max_tokens=3500,
        deterministic=False,
    )
    return result if isinstance(result, ExplanationBatch) else ExplanationBatch.model_validate(result)
