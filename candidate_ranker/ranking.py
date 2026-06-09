"""Local candidate feature engineering, scoring, and integrity analysis."""

from __future__ import annotations

import logging
import heapq
import re
from collections.abc import Iterable
from statistics import mean
from typing import Any

from candidate_ranker.models import CandidateScore, JDIntelligence, SchemaMap
from candidate_ranker.schema_mapping import display_name, resolve_paths
from candidate_ranker.text import normalize_text, skill_overlap, tokens

LOGGER = logging.getLogger(__name__)


def rank_candidates(
    candidates: list[dict[str, Any]],
    jd: JDIntelligence,
    schema_map: SchemaMap,
    top_k: int,
) -> list[CandidateScore]:
    """Rank candidates with local explainable scoring only."""

    scored = [_score_candidate(candidate, jd, schema_map) for candidate in candidates]
    scored.sort(key=lambda item: item.composite_score, reverse=True)
    LOGGER.info("Feature-scored %s candidates", len(scored))
    return scored[: min(top_k, len(scored))]


def rank_candidates_stream(
    candidates: Iterable[dict[str, Any]],
    jd: JDIntelligence,
    schema_map: SchemaMap,
    top_k: int,
) -> list[CandidateScore]:
    """Rank candidate streams while keeping only the best scored rows in memory."""

    if top_k <= 0:
        return []
    heap: list[tuple[float, int, CandidateScore]] = []
    seen = 0
    for seen, candidate in enumerate(candidates, start=1):
        normalized_text = normalize_text(candidate)
        candidate["_normalized_text"] = normalized_text
        candidate.setdefault("_retrieval_score", _lexical_retrieval_score(normalized_text, jd))
        score = _score_candidate(candidate, jd, schema_map)
        candidate.pop("_normalized_text", None)
        item = (score.composite_score, seen, score)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif score.composite_score > heap[0][0]:
            heapq.heapreplace(heap, item)
    ranked = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    LOGGER.info("Stream feature-scored %s candidates and kept top %s", seen, len(ranked))
    return ranked


def _score_candidate(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap) -> CandidateScore:
    all_text = str(candidate.get("_normalized_text") or normalize_text(candidate))
    desired_skills = [*jd.required_skills, *jd.preferred_skills]
    matched, missing = skill_overlap(all_text, desired_skills)
    required_matched, required_missing = skill_overlap(all_text, jd.required_skills)
    semantic = float(candidate.get("_retrieval_score", 0.0))
    skill_score = _ratio(len(matched), len(desired_skills))
    required_score = _ratio(len(required_matched), len(jd.required_skills))
    experience_score = _experience_score(candidate, jd, schema_map)
    production_score = _keyword_score(all_text, jd.production_signals + ["production", "scale", "deployed"])
    seniority_score = _keyword_score(all_text, [jd.seniority]) if jd.seniority else 0.5
    consistency_score, integrity_flags = _integrity(candidate, schema_map)
    education_score = _keyword_score(all_text, jd.education_keywords) if jd.education_keywords else 0.5
    location_score = _keyword_score(all_text, jd.locations) if jd.locations else 0.7
    behavioral_score = _keyword_score(all_text, jd.behavioral_signals) if jd.behavioral_signals else 0.5
    availability_score = 1.0 if resolve_paths(candidate, schema_map.availability.paths) else 0.5

    breakdown = {
        "semantic_match": _clamp((semantic + 1) / 2),
        "skill_overlap": skill_score,
        "required_skill_coverage": required_score,
        "skill_depth": min(1.0, len(matched) / 8),
        "relevant_experience": experience_score,
        "production_experience_signals": production_score,
        "seniority_match": seniority_score,
        "career_progression": consistency_score,
        "career_consistency": consistency_score,
        "behavioral_signals": behavioral_score,
        "availability": availability_score,
        "education_relevance": education_score,
        "location_match": location_score,
        "fraud_detection": 1.0 - min(1.0, len(integrity_flags) * 0.25),
        "consistency_analysis": consistency_score,
    }
    weights = {
        "semantic_match": 0.16,
        "skill_overlap": 0.11,
        "required_skill_coverage": 0.14,
        "skill_depth": 0.06,
        "relevant_experience": 0.11,
        "production_experience_signals": 0.06,
        "seniority_match": 0.06,
        "career_progression": 0.04,
        "career_consistency": 0.04,
        "behavioral_signals": 0.03,
        "availability": 0.03,
        "education_relevance": 0.04,
        "location_match": 0.04,
        "fraud_detection": 0.08,
        "consistency_analysis": 0.06,
    }
    composite = sum(breakdown[name] * weight for name, weight in weights.items()) * 100
    strengths = _strengths(breakdown, matched)
    concerns = _concerns(required_missing, integrity_flags, breakdown)
    return CandidateScore(
        candidate_id=_candidate_id(candidate, schema_map),
        display_name=display_name(candidate, schema_map),
        composite_score=round(composite, 2),
        breakdown={key: round(value * 100, 2) for key, value in breakdown.items()},
        matched_skills=matched,
        missing_skills=missing,
        strengths=strengths,
        concerns=concerns,
        integrity_flags=integrity_flags,
        raw_candidate=candidate,
    )


def _experience_score(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap) -> float:
    values = resolve_paths(candidate, schema_map.experience.paths)
    text = normalize_text(values or candidate)
    numbers = [float(match) for match in re.findall(r"\b\d+(?:\.\d+)?\b", text)]
    candidate_years = max(numbers) if numbers else 0.0
    if jd.experience_years_min is None:
        return min(1.0, candidate_years / 8) if candidate_years else 0.5
    return _clamp(candidate_years / max(jd.experience_years_min, 1.0))


def _candidate_id(candidate: dict[str, Any], schema_map: SchemaMap) -> str:
    values = [
        candidate.get("candidate_id"),
        candidate.get("id"),
        *resolve_paths(candidate, [path for path in schema_map.identity.paths if path.lower().endswith("_id")]),
        candidate.get("_row_id"),
    ]
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float):
            return str(value)
    return display_name(candidate, schema_map)


def _keyword_score(text: str, keywords: list[str]) -> float:
    keyword_tokens = [keyword for keyword in keywords if keyword.strip()]
    if not keyword_tokens:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for keyword in keyword_tokens if keyword.lower() in lowered)
    return _ratio(hits, len(keyword_tokens))


def _lexical_retrieval_score(candidate_text: str, jd: JDIntelligence) -> float:
    text = candidate_text.lower()
    terms = [
        *jd.required_skills,
        *jd.preferred_skills,
        jd.seniority,
        *jd.production_signals,
        *jd.responsibilities[:8],
    ]
    keywords = [term.lower() for term in terms if term and term.strip()]
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in text)
    return (2 * _ratio(hits, len(keywords))) - 1


def _integrity(candidate: dict[str, Any], schema_map: SchemaMap) -> tuple[float, list[str]]:
    text = normalize_text(candidate)
    flags: list[str] = []
    if len(text) < 80:
        flags.append("Sparse profile data")
    if len(tokens(text)) < 20:
        flags.append("Low information diversity")
    experience_values = normalize_text(resolve_paths(candidate, schema_map.experience.paths))
    if experience_values and "present" in experience_values.lower() and not re.search(r"\b20\d{2}\b", experience_values):
        flags.append("Open-ended experience dates need review")
    duplicate_signal = mean([text.lower().count(term) for term in ("expert", "guru", "rockstar")])
    if duplicate_signal > 3:
        flags.append("Repeated promotional language")
    return max(0.0, 1.0 - 0.2 * len(flags)), flags


def _strengths(breakdown: dict[str, float], matched: list[str]) -> list[str]:
    labels = {
        "skill_overlap": "Strong skill alignment",
        "required_skill_coverage": "Required skills covered",
        "skill_depth": "Broad relevant skill depth",
        "relevant_experience": "Extensive professional experience",
        "production_experience_signals": "Production delivery signals",
        "seniority_match": "Senior-level background",
        "career_progression": "Consistent career progression",
        "career_consistency": "Consistent career history",
        "behavioral_signals": "Relevant behavioral signals",
        "availability": "Availability information present",
        "education_relevance": "Relevant education background",
        "location_match": "Location alignment",
        "fraud_detection": "Low integrity risk",
        "consistency_analysis": "Consistent profile evidence",
    }
    strengths = [
        labels.get(name, name.replace("_", " ").title())
        for name, value in breakdown.items()
        if value >= 0.75 and name != "semantic_match"
    ]
    if matched:
        strengths.insert(0, f"Matches {len(matched)} desired skills")
    return strengths[:6]


def _concerns(missing: list[str], flags: list[str], breakdown: dict[str, float]) -> list[str]:
    concerns = [f"Missing required skill: {skill}" for skill in missing[:4]]
    concerns.extend(flags)
    concerns.extend(
        f"Weak {name.replace('_', ' ')}" for name, value in breakdown.items() if value <= 0.25
    )
    return concerns[:8]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else _clamp(numerator / denominator)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
