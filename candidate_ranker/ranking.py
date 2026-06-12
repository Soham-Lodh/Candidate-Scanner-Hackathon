"""Local candidate feature engineering, scoring, and integrity analysis."""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections.abc import Iterable
from statistics import mean
from typing import Any

from candidate_ranker.models import CandidateScore, JDIntelligence, SchemaMap
from candidate_ranker.schema_mapping import display_name, resolve_paths
from candidate_ranker.text import normalize_text, skill_overlap, tokens

LOGGER = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "for",
    "hands",
    "in",
    "of",
    "on",
    "or",
    "role",
    "systems",
    "the",
    "to",
    "with",
    "work",
}

GENERIC_SKILL_WORDS = {
    "advanced",
    "building",
    "design",
    "designing",
    "development",
    "experience",
    "expert",
    "excellent",
    "hands",
    "infrastructure",
    "production",
    "proficient",
    "strong",
}

PROFICIENCY_WEIGHTS = {
    "beginner": 0.45,
    "intermediate": 0.68,
    "advanced": 0.86,
    "expert": 1.0,
}


def rank_candidates(
    candidates: list[dict[str, Any]],
    jd: JDIntelligence,
    schema_map: SchemaMap,
    top_k: int,
) -> list[CandidateScore]:
    """Rank candidates with local explainable scoring only."""

    scored = [_score_candidate(candidate, jd, schema_map) for candidate in candidates]
    scored.sort(key=lambda item: (-item.composite_score, item.candidate_id))
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
    heap: list[tuple[float, str, int, CandidateScore]] = []
    seen = 0
    for seen, candidate in enumerate(candidates, start=1):
        normalized_text = normalize_text(candidate)
        candidate["_normalized_text"] = normalized_text
        candidate.setdefault("_retrieval_score", _lexical_retrieval_score(normalized_text, jd))
        score = _score_candidate(candidate, jd, schema_map)
        candidate.pop("_normalized_text", None)
        item = (score.composite_score, score.candidate_id, seen, score)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    ranked = [item[3] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    LOGGER.info("Stream feature-scored %s candidates and kept top %s", seen, len(ranked))
    return ranked


def _score_candidate(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap) -> CandidateScore:
    all_text = str(candidate.get("_normalized_text") or normalize_text(candidate))
    all_tokens = tokens(all_text)
    lowered_text = all_text.lower()
    skill_records = _skill_records(candidate, schema_map)
    required = _clean_terms(jd.required_skills)
    preferred = _clean_terms(jd.preferred_skills)
    desired = [*required, *preferred]

    required_scores = [_term_match_score(term, lowered_text, all_tokens, skill_records) for term in required]
    preferred_scores = [_term_match_score(term, lowered_text, all_tokens, skill_records) for term in preferred]
    desired_scores = [*required_scores, *preferred_scores]

    required_coverage = _weighted_mean(required_scores) if required_scores else 0.58
    preferred_coverage = _weighted_mean(preferred_scores) if preferred_scores else 0.55
    skill_overlap_score = _weighted_mean(desired_scores) if desired_scores else _profile_skill_depth(skill_records)
    skill_depth = _skill_depth_score(skill_records, desired, all_text)
    matched_skills = [
        term
        for term, score in zip(desired, desired_scores, strict=False)
        if score >= 0.52
    ]
    missing_required = [
        term
        for term, score in zip(required, required_scores, strict=False)
        if score < 0.42
    ]
    missing_skills = [
        term
        for term, score in zip(desired, desired_scores, strict=False)
        if score < 0.42
    ]

    semantic_match = _semantic_score(float(candidate.get("_retrieval_score", 0.0)))
    lexical_relevance = _lexical_relevance_score(all_tokens, jd)
    role_fit = _role_fit_score(candidate, jd, schema_map, all_text, all_tokens)
    experience_score, years = _experience_score(candidate, jd, schema_map)
    production_score = _production_score(candidate, jd, all_text, all_tokens)
    seniority_score = _seniority_score(candidate, jd, years, all_text)
    career_score = _career_consistency_score(candidate, schema_map, all_text)
    behavioral_score = _behavioral_score(candidate)
    availability_score = _availability_score(candidate, schema_map)
    education_score = _education_score(candidate, jd, schema_map, all_text)
    location_score = _location_score(candidate, jd, schema_map, all_text)
    integrity_score, integrity_flags = _integrity(candidate, schema_map, all_text, all_tokens)
    evidence_confidence = _evidence_confidence(candidate, all_tokens, skill_records)

    technical_fit = (
        0.42 * required_coverage
        + 0.20 * preferred_coverage
        + 0.18 * skill_depth
        + 0.12 * production_score
        + 0.08 * lexical_relevance
    )
    role_relevance = (
        0.38 * role_fit
        + 0.24 * semantic_match
        + 0.20 * experience_score
        + 0.10 * seniority_score
        + 0.08 * career_score
    )
    hireability = (
        0.30 * behavioral_score
        + 0.23 * availability_score
        + 0.18 * location_score
        + 0.16 * education_score
        + 0.13 * evidence_confidence
    )
    must_have_gate = 0.52 * required_coverage + 0.22 * role_fit + 0.16 * experience_score + 0.10 * integrity_score
    base_score = 0.55 * technical_fit + 0.32 * role_relevance + 0.13 * hireability
    calibrated = base_score * (0.70 + 0.30 * must_have_gate) * (0.84 + 0.16 * integrity_score)
    composite = _clamp(calibrated) * 100

    breakdown = {
        "semantic_match": semantic_match,
        "lexical_relevance": lexical_relevance,
        "technical_fit": technical_fit,
        "skill_overlap": skill_overlap_score,
        "required_skill_coverage": required_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "skill_depth": skill_depth,
        "role_fit": role_fit,
        "relevant_experience": experience_score,
        "production_experience_signals": production_score,
        "seniority_match": seniority_score,
        "career_progression": career_score,
        "career_consistency": career_score,
        "behavioral_signals": behavioral_score,
        "availability": availability_score,
        "education_relevance": education_score,
        "location_match": location_score,
        "fraud_detection": integrity_score,
        "consistency_analysis": career_score,
        "evidence_confidence": evidence_confidence,
        "must_have_gate": must_have_gate,
    }
    strengths = _strengths(breakdown, matched_skills)
    concerns = _concerns(missing_required, integrity_flags, breakdown)
    return CandidateScore(
        candidate_id=_candidate_id(candidate, schema_map),
        display_name=display_name(candidate, schema_map),
        composite_score=round(composite, 4),
        breakdown={key: round(value * 100, 2) for key, value in breakdown.items()},
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        strengths=strengths,
        concerns=concerns,
        integrity_flags=integrity_flags,
        raw_candidate=candidate,
    )


def _skill_records(candidate: dict[str, Any], schema_map: SchemaMap) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_skills = candidate.get("skills")
    if isinstance(raw_skills, list):
        for item in raw_skills:
            if isinstance(item, dict) and item.get("name"):
                records.append(item)
            elif isinstance(item, str):
                records.append({"name": item})
    if records:
        return records
    for value in resolve_paths(candidate, schema_map.skills.paths):
        if isinstance(value, dict) and value.get("name"):
            records.append(value)
        elif isinstance(value, str):
            records.append({"name": value})
    return records


def _term_match_score(
    term: str,
    lowered_text: str,
    text_tokens: set[str],
    skill_records: list[dict[str, Any]],
) -> float:
    term_tokens = _signal_tokens(term, drop_generic=True)
    if not term_tokens:
        return 0.0
    exact_bonus = 0.16 if term.lower() in lowered_text else 0.0
    text_coverage = len(term_tokens & text_tokens) / len(term_tokens)
    text_score = min(1.0, (text_coverage**1.35) + exact_bonus)

    skill_score = 0.0
    for record in skill_records:
        name = str(record.get("name") or "")
        name_tokens = _signal_tokens(name, drop_generic=True)
        if not name_tokens:
            continue
        overlap = len(term_tokens & name_tokens)
        if not overlap:
            continue
        precision = overlap / len(name_tokens)
        recall = overlap / len(term_tokens)
        token_score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        evidence = _skill_evidence_score(record)
        skill_score = max(skill_score, min(1.0, token_score * (0.78 + 0.22 * evidence)))
    return _clamp(max(text_score, skill_score))


def _skill_evidence_score(record: dict[str, Any]) -> float:
    proficiency = PROFICIENCY_WEIGHTS.get(str(record.get("proficiency") or "").lower(), 0.62)
    duration = _number(record.get("duration_months"))
    endorsements = _number(record.get("endorsements"))
    duration_score = _soft_cap(duration or 0.0, 48.0)
    endorsement_score = _soft_cap(endorsements or 0.0, 40.0)
    return _clamp(0.62 * proficiency + 0.26 * duration_score + 0.12 * endorsement_score)


def _skill_depth_score(skill_records: list[dict[str, Any]], desired: list[str], text: str) -> float:
    if not skill_records:
        if not desired:
            return 0.45
        matches, _ = skill_overlap(text, desired)
        return _soft_cap(len(matches), 8.0)
    evidence_scores = sorted((_skill_evidence_score(record) for record in skill_records), reverse=True)
    top_evidence = mean(evidence_scores[: min(6, len(evidence_scores))])
    breadth = _soft_cap(len(skill_records), 14.0)
    return _clamp(0.58 * top_evidence + 0.42 * breadth)


def _profile_skill_depth(skill_records: list[dict[str, Any]]) -> float:
    if not skill_records:
        return 0.35
    return _skill_depth_score(skill_records, [], "")


def _role_fit_score(
    candidate: dict[str, Any],
    jd: JDIntelligence,
    schema_map: SchemaMap,
    text: str,
    text_tokens: set[str],
) -> float:
    role_sources = [
        _path_text(candidate, "profile.current_title"),
        _path_text(candidate, "profile.headline"),
        normalize_text(resolve_paths(candidate, schema_map.seniority.paths)),
        normalize_text(resolve_paths(candidate, schema_map.projects.paths)),
    ]
    role_text = " ".join(item for item in role_sources if item).strip() or text
    jd_role_terms = [jd.title, jd.seniority, *jd.responsibilities[:6]]
    query_tokens = set().union(*[_signal_tokens(term) for term in jd_role_terms if term])
    if not query_tokens:
        return 0.58
    role_tokens = tokens(role_text)
    title_overlap = len(query_tokens & role_tokens) / len(query_tokens)
    profile_overlap = len(query_tokens & text_tokens) / len(query_tokens)
    return _clamp(0.70 * (title_overlap**0.8) + 0.30 * (profile_overlap**0.9))


def _experience_score(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap) -> tuple[float, float]:
    years = _candidate_years(candidate, schema_map)
    if years <= 0:
        return 0.45, 0.0
    minimum = jd.experience_years_min
    if minimum is None or minimum <= 0:
        return _clamp(0.30 + 0.70 * _soft_cap(years, 8.0)), years
    if years >= minimum:
        surplus = years - minimum
        return _clamp(0.74 + 0.26 * _soft_cap(surplus, 6.0)), years
    return _clamp(0.74 * ((years / minimum) ** 1.45)), years


def _candidate_years(candidate: dict[str, Any], schema_map: SchemaMap) -> float:
    direct = _path_number(candidate, "profile.years_of_experience")
    if direct is not None:
        return direct
    candidates: list[float] = []
    for path in schema_map.experience.paths:
        lowered = path.lower()
        if "year" not in lowered or "date" in lowered or "start_year" in lowered or "end_year" in lowered:
            continue
        for value in resolve_paths(candidate, [path]):
            number = _number(value)
            if number is not None and 0 <= number <= 50:
                candidates.append(number)
    if candidates:
        return max(candidates)
    durations = []
    for value in _iter_key_values(candidate, "duration_months"):
        number = _number(value)
        if number is not None and number >= 0:
            durations.append(number)
    return sum(durations) / 12.0 if durations else 0.0


def _production_score(candidate: dict[str, Any], jd: JDIntelligence, text: str, text_tokens: set[str]) -> float:
    production_terms = [
        *jd.production_signals,
        "production",
        "deployed",
        "scale",
        "scalable",
        "latency",
        "throughput",
        "pipeline",
        "monitoring",
        "on-call",
        "kubernetes",
        "docker",
        "ci/cd",
        "aws",
        "gcp",
        "azure",
    ]
    keyword_score = _keyword_density_score(text_tokens, production_terms)
    metric_score = min(1.0, len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|gb|tb|ms|k|m|users?|requests?|records?)\b", text.lower())) / 5)
    return _clamp(0.78 * keyword_score + 0.22 * metric_score)


def _seniority_score(candidate: dict[str, Any], jd: JDIntelligence, years: float, text: str) -> float:
    seniority = jd.seniority.lower().strip()
    title = " ".join([_path_text(candidate, "profile.current_title"), _path_text(candidate, "profile.headline")]).lower()
    if not seniority:
        return _clamp(0.35 + 0.65 * _soft_cap(years, 8.0))
    senior_terms = {"senior", "lead", "principal", "staff", "manager", "head", "architect"}
    junior_terms = {"junior", "entry", "associate", "intern", "trainee"}
    if any(term in seniority for term in senior_terms):
        title_signal = 1.0 if any(term in title for term in senior_terms) else 0.0
        return _clamp(0.28 * title_signal + 0.72 * _soft_cap(years, 7.0))
    if any(term in seniority for term in junior_terms):
        return _clamp(1.0 - min(0.55, max(0.0, years - 4.0) / 12.0))
    return _keyword_density_score(tokens(title or text), [seniority])


def _career_consistency_score(candidate: dict[str, Any], schema_map: SchemaMap, text: str) -> float:
    history = candidate.get("career_history")
    if not isinstance(history, list) or not history:
        return _clamp(0.45 + 0.35 * _soft_cap(len(tokens(text)), 160.0))
    durations = [_number(item.get("duration_months")) for item in history if isinstance(item, dict)]
    durations = [value for value in durations if value is not None]
    avg_duration = mean(durations) if durations else 0.0
    tenure_score = _clamp(0.35 + 0.65 * _soft_cap(avg_duration, 30.0))
    current_title = _path_text(candidate, "profile.current_title").lower()
    current_history = " ".join(
        str(item.get("title") or "").lower()
        for item in history
        if isinstance(item, dict) and item.get("is_current")
    )
    title_score = 1.0 if current_title and current_title in current_history else 0.72
    return _clamp(0.72 * tenure_score + 0.28 * title_score)


def _behavioral_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals")
    if not isinstance(signals, dict):
        return 0.55
    response = _number(signals.get("recruiter_response_rate"))
    response_time = _number(signals.get("avg_response_time_hours"))
    completeness = _number(signals.get("profile_completeness_score"))
    views = _number(signals.get("profile_views_received_30d"))
    saved = _number(signals.get("saved_by_recruiters_30d"))
    completion = _number(signals.get("interview_completion_rate"))
    offer = _number(signals.get("offer_acceptance_rate"))
    github = _number(signals.get("github_activity_score"))
    verification = mean(
        [
            1.0 if signals.get("verified_email") else 0.0,
            1.0 if signals.get("verified_phone") else 0.0,
            1.0 if signals.get("linkedin_connected") else 0.0,
        ]
    )
    components = [
        (response if response is not None else 0.45, 0.24),
        (1.0 - _soft_cap(response_time or 168.0, 168.0), 0.10),
        (_clamp((completeness or 55.0) / 100), 0.14),
        (_soft_cap(views or 0.0, 80.0), 0.10),
        (_soft_cap(saved or 0.0, 12.0), 0.10),
        (completion if completion is not None else 0.58, 0.14),
        ((offer if offer is not None and offer >= 0 else 0.55), 0.08),
        ((_clamp(github / 100) if github is not None and github >= 0 else 0.45), 0.05),
        (verification, 0.05),
    ]
    return _clamp(sum(value * weight for value, weight in components))


def _availability_score(candidate: dict[str, Any], schema_map: SchemaMap) -> float:
    signals = candidate.get("redrob_signals")
    if isinstance(signals, dict):
        notice = _number(signals.get("notice_period_days"))
        open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.48
        notice_score = 1.0 - _soft_cap(notice or 90.0, 120.0)
        relocate = 1.0 if signals.get("willing_to_relocate") else 0.62
        return _clamp(0.48 * open_to_work + 0.34 * notice_score + 0.18 * relocate)
    return 0.65 if resolve_paths(candidate, schema_map.availability.paths) else 0.48


def _education_score(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap, text: str) -> float:
    education_text = normalize_text(candidate.get("education") or resolve_paths(candidate, schema_map.education.paths))
    if not education_text:
        return 0.48
    keyword_score = _keyword_density_score(tokens(education_text), jd.education_keywords) if jd.education_keywords else 0.55
    tier_bonus = 0.0
    lowered = education_text.lower()
    if "tier_1" in lowered:
        tier_bonus = 0.25
    elif "tier_2" in lowered:
        tier_bonus = 0.16
    elif "tier_3" in lowered:
        tier_bonus = 0.08
    degree_bonus = 0.10 if re.search(r"\b(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|phd|computer science|data science)\b", lowered) else 0.0
    return _clamp(keyword_score + tier_bonus + degree_bonus)


def _location_score(candidate: dict[str, Any], jd: JDIntelligence, schema_map: SchemaMap, text: str) -> float:
    if not jd.locations:
        return 0.68
    location_text = " ".join(
        [
            _path_text(candidate, "profile.location"),
            _path_text(candidate, "profile.country"),
            normalize_text(resolve_paths(candidate, schema_map.location.paths)),
        ]
    )
    if not location_text.strip():
        location_text = text
    score = _keyword_density_score(tokens(location_text), jd.locations)
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate") if isinstance(candidate.get("redrob_signals"), dict) else None
    if relocate:
        score = max(score, 0.72)
    return _clamp(score)


def _lexical_relevance_score(text_tokens: set[str], jd: JDIntelligence) -> float:
    terms = [*jd.required_skills, *jd.preferred_skills, jd.title, *jd.responsibilities[:8]]
    return _keyword_density_score(text_tokens, terms)


def _lexical_retrieval_score(candidate_text: str, jd: JDIntelligence) -> float:
    relevance = _lexical_relevance_score(tokens(candidate_text), jd)
    return (2 * relevance) - 1


def _semantic_score(raw_score: float) -> float:
    if -1.0 <= raw_score <= 1.0:
        return _clamp((raw_score + 1.0) / 2.0)
    return _clamp(raw_score)


def _integrity(
    candidate: dict[str, Any],
    schema_map: SchemaMap,
    text: str,
    text_tokens: set[str],
) -> tuple[float, list[str]]:
    flags: list[str] = []
    token_count = len(text_tokens)
    if len(text) < 80:
        flags.append("Sparse profile data")
    if token_count < 20:
        flags.append("Low information diversity")
    completeness = _path_number(candidate, "redrob_signals.profile_completeness_score")
    if completeness is not None and completeness < 45:
        flags.append("Low profile completeness")
    experience_values = normalize_text(resolve_paths(candidate, schema_map.experience.paths))
    if experience_values and "present" in experience_values.lower() and not re.search(r"\b20\d{2}\b", experience_values):
        flags.append("Open-ended experience dates need review")
    duplicate_signal = mean([text.lower().count(term) for term in ("expert", "guru", "rockstar")])
    if duplicate_signal > 3:
        flags.append("Repeated promotional language")
    score = 1.0 - min(0.62, 0.16 * len(flags))
    score *= 0.86 + 0.14 * _soft_cap(token_count, 180.0)
    return _clamp(score), flags


def _evidence_confidence(candidate: dict[str, Any], text_tokens: set[str], skill_records: list[dict[str, Any]]) -> float:
    profile = candidate.get("profile") if isinstance(candidate.get("profile"), dict) else {}
    history = candidate.get("career_history") if isinstance(candidate.get("career_history"), list) else []
    education = candidate.get("education") if isinstance(candidate.get("education"), list) else []
    signals = candidate.get("redrob_signals") if isinstance(candidate.get("redrob_signals"), dict) else {}
    components = [
        1.0 if profile else 0.0,
        _soft_cap(len(skill_records), 12.0),
        _soft_cap(len(history), 4.0),
        _soft_cap(len(education), 2.0),
        _soft_cap(len(text_tokens), 220.0),
        _soft_cap(len(signals), 18.0),
    ]
    return _clamp(mean(components))


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


def _strengths(breakdown: dict[str, float], matched: list[str]) -> list[str]:
    labels = {
        "technical_fit": "Strong technical fit",
        "skill_overlap": "Strong skill alignment",
        "required_skill_coverage": "Required skills covered",
        "preferred_skill_coverage": "Preferred skills covered",
        "skill_depth": "Deep skill evidence",
        "role_fit": "Relevant current role",
        "relevant_experience": "Experience level fits role",
        "production_experience_signals": "Production delivery evidence",
        "seniority_match": "Seniority aligned",
        "career_progression": "Consistent career progression",
        "career_consistency": "Consistent career history",
        "behavioral_signals": "Strong engagement signals",
        "availability": "Favorable availability",
        "education_relevance": "Relevant education background",
        "location_match": "Location alignment",
        "fraud_detection": "Low integrity risk",
        "evidence_confidence": "Rich profile evidence",
    }
    strengths = [
        labels.get(name, name.replace("_", " ").title())
        for name, value in breakdown.items()
        if value >= 0.72 and name not in {"semantic_match", "must_have_gate"}
    ]
    if matched:
        strengths.insert(0, f"Matches {len(matched)} desired skills")
    return strengths[:7]


def _concerns(missing: list[str], flags: list[str], breakdown: dict[str, float]) -> list[str]:
    concerns = [f"Missing required skill: {skill}" for skill in missing[:4]]
    concerns.extend(flags)
    concern_labels = {
        "technical_fit": "Weak technical fit",
        "required_skill_coverage": "Low required-skill coverage",
        "role_fit": "Current role appears weakly aligned",
        "relevant_experience": "Experience below role target",
        "production_experience_signals": "Limited production evidence",
        "behavioral_signals": "Weak engagement signals",
        "availability": "Availability may be less favorable",
        "evidence_confidence": "Profile evidence is thin",
    }
    for name, label in concern_labels.items():
        if breakdown.get(name, 1.0) <= 0.32 and label not in concerns:
            concerns.append(label)
    return concerns[:8]


def _keyword_density_score(text_tokens: set[str], keywords: list[str]) -> float:
    terms = [_signal_tokens(keyword) for keyword in keywords if keyword and keyword.strip()]
    terms = [term for term in terms if term]
    if not terms:
        return 0.0
    if not text_tokens:
        return 0.0
    scores = []
    for term in terms:
        coverage = len(term & text_tokens) / len(term)
        scores.append(min(1.0, coverage**1.2))
    positive = [score for score in scores if score > 0]
    if not positive:
        return 0.0
    breadth = len(positive) / len(scores)
    depth = mean(sorted(positive, reverse=True)[: min(6, len(positive))])
    return _clamp(0.62 * depth + 0.38 * (breadth**0.7))


def _signal_tokens(value: str, *, drop_generic: bool = False) -> set[str]:
    found = tokens(value)
    found = {token for token in found if token not in STOPWORDS and len(token) > 1}
    if drop_generic:
        found = {token for token in found if token not in GENERIC_SKILL_WORDS}
    return found


def _clean_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = re.sub(r"\s+", " ", term.strip())
        key = normalized.lower()
        if normalized and key not in seen:
            cleaned.append(normalized)
            seen.add(key)
    return cleaned


def _weighted_mean(values: list[float]) -> float:
    return _clamp(sum(values) / len(values)) if values else 0.0


def _soft_cap(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return _clamp(1.0 - math.exp(-max(0.0, value) / scale))


def _path_text(candidate: dict[str, Any], path: str) -> str:
    current: Any = candidate
    for part in path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    return normalize_text(current)


def _path_number(candidate: dict[str, Any], path: str) -> float | None:
    current: Any = candidate
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _number(current)


def _iter_key_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from _iter_key_values(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_key_values(item, key)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
