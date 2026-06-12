"""CSV export helpers for ranked candidate results."""

from __future__ import annotations

import logging
import re
from io import StringIO
from typing import Any

import pandas as pd

from candidate_ranker.models import CandidateExplanation, CandidateScore

LOGGER = logging.getLogger(__name__)


TOP_CANDIDATE_EXPORT_LIMIT = 100


def rankings_to_csv(
    scores: list[CandidateScore],
    explanations: list[CandidateExplanation] | None = None,
    limit: int = TOP_CANDIDATE_EXPORT_LIMIT,
) -> str:
    """Serialize top ranked candidates to CSV for recruiter export."""

    rows = ranking_export_rows(scores, explanations, limit)
    buffer = StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    LOGGER.info("Exported %s ranked candidates to CSV", len(rows))
    return buffer.getvalue()


def ranking_export_rows(
    scores: list[CandidateScore],
    explanations: list[CandidateExplanation] | None = None,
    limit: int = TOP_CANDIDATE_EXPORT_LIMIT,
) -> list[dict[str, object]]:
    """Build display/export rows for the top ranked candidates."""

    ordered_scores = sorted(scores, key=lambda score: (-score.composite_score, score.candidate_id))
    export_scores = ordered_scores[:limit]
    normalized_scores = _normalized_export_scores(export_scores)
    return [
        {
            "candidate_id": score.candidate_id,
            "rank": index + 1,
            "score": f"{normalized_scores[index]:.4f}",
            "reasoning": scoring_reasoning(score),
        }
        for index, score in enumerate(export_scores)
    ]


def _normalized_export_scores(scores: list[CandidateScore]) -> list[float]:
    """Min-max normalize only the scores shown in the exported CSV."""

    if not scores:
        return []
    raw_scores = [score.composite_score for score in scores]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score == min_score:
        return [1.0 for _ in raw_scores]
    return [
        0.1 + 0.9 * ((score - min_score) / (max_score - min_score))
        for score in raw_scores
    ]


def scoring_reasoning(score: CandidateScore) -> str:
    """Explain ranking using local scoring evidence in recruiter-ready language."""

    evidence_reasoning = _candidate_evidence_reasoning(score)
    if evidence_reasoning:
        return evidence_reasoning

    opening = _opening_assessment(score)
    strengths = _strength_bullets(score)
    concerns = _concern_bullets(score)
    overall = _overall_assessment(score, concerns)
    return "\n\n".join(
        [
            opening,
            "Key strengths:\n" + "\n".join(f"- {item}" for item in strengths),
            "Potential concerns:\n" + "\n".join(f"- {item}" for item in concerns),
            "Overall assessment:\n" + overall,
        ]
    )


def _opening_assessment(score: CandidateScore) -> str:
    if score.matched_skills and score.breakdown.get("relevant_experience", 0) >= 75:
        skill = _skill_label(score.matched_skills[0])
        return (
            f"Candidate demonstrates strong {skill} expertise "
            "and extensive professional experience."
        )
    if score.matched_skills:
        return f"Candidate demonstrates relevant {_skill_label(score.matched_skills[0])} capability for this role."
    return "Candidate shows partial alignment with the role based on the available profile evidence."


def _strength_bullets(score: CandidateScore) -> list[str]:
    strengths: list[str] = []
    breakdown = score.breakdown
    if breakdown.get("seniority_match", 0) >= 75:
        strengths.append("Senior-level engineering background")
    if breakdown.get("career_progression", 0) >= 75 or breakdown.get("career_consistency", 0) >= 75:
        strengths.append("Consistent career progression")
    if breakdown.get("required_skill_coverage", 0) >= 70 or breakdown.get("skill_overlap", 0) >= 70:
        strengths.append("Strong alignment with core role requirements")
    if breakdown.get("production_experience_signals", 0) >= 60:
        strengths.append("Evidence of production delivery experience")
    if score.matched_skills:
        strengths.append(f"Relevant skills include {', '.join(score.matched_skills[:5])}")
    for strength in score.strengths:
        if strength not in strengths and "semantic" not in strength.lower():
            strengths.append(strength)
    return strengths[:4] or ["Best available match among the ranked candidate pool"]


def _concern_bullets(score: CandidateScore) -> list[str]:
    concerns: list[str] = []
    for skill in score.missing_skills[:5]:
        concerns.append(f"No evidence of {skill} experience")
    for concern in score.concerns:
        cleaned = concern.removeprefix("Missing required skill: ").strip()
        if cleaned and cleaned != concern:
            item = f"No evidence of {cleaned} experience"
        else:
            item = concern
        if item not in concerns:
            concerns.append(item)
    return concerns[:4] or ["No major concerns surfaced from the scoring evidence"]


def _overall_assessment(score: CandidateScore, concerns: list[str]) -> str:
    if score.composite_score >= 80 and concerns:
        return (
            "Strong candidate, but may require targeted onboarding for the missing "
            "responsibilities noted above."
        )
    if score.composite_score >= 70:
        return "Strong candidate with solid evidence for the role requirements."
    if score.composite_score >= 55:
        return "Moderate match; worth reviewing if the team can support skill ramp-up."
    return "Lower-confidence match based on the current job criteria and profile evidence."


def _skill_label(skill: str) -> str:
    return re.sub(r"^(strong|expert|advanced|excellent)\s+", "", skill.strip(), flags=re.IGNORECASE)


def _candidate_evidence_reasoning(score: CandidateScore) -> str:
    candidate = score.raw_candidate
    profile = _dict_value(candidate.get("profile"))
    signals = _dict_value(candidate.get("redrob_signals"))
    skills = [item for item in candidate.get("skills", []) if isinstance(item, dict)]
    if not profile and not signals and not skills:
        return ""

    title = str(profile.get("current_title") or profile.get("headline") or "Candidate").strip()
    years = _number(profile.get("years_of_experience"))
    location = ", ".join(
        part
        for part in [
            str(profile.get("location") or "").strip(),
            str(profile.get("country") or "").strip(),
        ]
        if part
    )
    matched_skills = _matched_skill_records(skills, score.matched_skills)
    skill_phrase = _skill_phrase(matched_skills, skills)
    assessment_phrase = _assessment_phrase(signals)
    signal_phrase = _signal_phrase(signals)
    gap_phrase = _gap_phrase(score)

    lead_parts = [title]
    if years is not None:
        lead_parts.append(f"{years:.1f} yrs")
    if location:
        lead_parts.append(location)

    evidence = [skill_phrase, assessment_phrase, signal_phrase, gap_phrase]
    evidence = [item for item in evidence if item]
    if not evidence:
        return ""
    return "; ".join([" with ".join(lead_parts[:2]) if len(lead_parts) > 1 else lead_parts[0], *evidence]) + "."


def _matched_skill_records(skills: list[dict[str, Any]], matched: list[str]) -> list[dict[str, Any]]:
    if not skills:
        return []
    wanted = [_canonical_skill(skill) for skill in matched]
    records: list[dict[str, Any]] = []
    for skill in skills:
        name = _canonical_skill(str(skill.get("name") or ""))
        if not name:
            continue
        if any(name in item or item in name for item in wanted):
            records.append(skill)
    return _sort_skills(records)


def _skill_phrase(matched_skills: list[dict[str, Any]], all_skills: list[dict[str, Any]]) -> str:
    selected = matched_skills[:3] or _sort_skills(all_skills)[:3]
    if not selected:
        return ""
    names = [_skill_record_label(skill) for skill in selected if skill.get("name")]
    ai_count = sum(1 for skill in all_skills if _is_ai_skill(str(skill.get("name") or "")))
    if matched_skills:
        prefix = f"{len(matched_skills)} matched JD skills"
    elif ai_count:
        prefix = f"{ai_count} AI/data skills"
    else:
        prefix = "top profile skills"
    return f"{prefix}: {', '.join(names)}"


def _skill_record_label(skill: dict[str, Any]) -> str:
    name = str(skill.get("name") or "").strip()
    proficiency = str(skill.get("proficiency") or "").strip()
    duration = _number(skill.get("duration_months"))
    endorsements = _number(skill.get("endorsements"))
    details: list[str] = []
    if proficiency:
        details.append(proficiency)
    if duration is not None and duration >= 12:
        details.append(f"{int(duration)} mo")
    if endorsements is not None and endorsements > 0:
        details.append(f"{int(endorsements)} endorsements")
    return f"{name} ({', '.join(details)})" if details else name


def _assessment_phrase(signals: dict[str, Any]) -> str:
    assessments = _dict_value(signals.get("skill_assessment_scores"))
    scores = [_number(value) for value in assessments.values()]
    scores = [value for value in scores if value is not None]
    if not scores:
        return ""
    average = sum(scores) / len(scores)
    return f"{len(scores)} skill assessments averaging {average:.0f}/100"


def _signal_phrase(signals: dict[str, Any]) -> str:
    if not signals:
        return ""
    response = _number(signals.get("recruiter_response_rate"))
    notice = _number(signals.get("notice_period_days"))
    mode = str(signals.get("preferred_work_mode") or "").strip()
    github = _number(signals.get("github_activity_score"))
    parts: list[str] = []
    if response is not None:
        parts.append(f"response rate {response:.2f}")
    if notice is not None:
        parts.append(f"{int(notice)}d notice")
    if mode:
        parts.append(f"{mode} work preference")
    if github is not None and github >= 0:
        parts.append(f"GitHub activity {github:.0f}/100")
    return ", ".join(parts[:3])


def _gap_phrase(score: CandidateScore) -> str:
    if not score.missing_skills:
        return "no major JD skill gaps flagged"
    missing = [_skill_label(skill) for skill in score.missing_skills[:2]]
    return "gaps: " + ", ".join(missing)


def _sort_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    return sorted(
        skills,
        key=lambda skill: (
            weights.get(str(skill.get("proficiency") or "").lower(), 0),
            _number(skill.get("duration_months")) or 0,
            _number(skill.get("endorsements")) or 0,
        ),
        reverse=True,
    )


def _is_ai_skill(name: str) -> bool:
    lowered = name.lower()
    keywords = (
        "ai",
        "ml",
        "llm",
        "nlp",
        "rag",
        "lora",
        "qlora",
        "peft",
        "bert",
        "gpt",
        "python",
        "pytorch",
        "tensorflow",
        "sklearn",
        "transformer",
        "embedding",
        "vector",
        "milvus",
        "pinecone",
        "faiss",
        "langchain",
        "machine learning",
        "deep learning",
        "fine-tuning",
        "feature engineering",
        "model",
    )
    return any(keyword in lowered for keyword in keywords)


def _canonical_skill(skill: str) -> str:
    cleaned = _skill_label(skill).lower()
    cleaned = re.sub(r"\b(experience|hands-on|designing|production|with|based|systems|infrastructure)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9+#.]+", " ", cleaned)
    return " ".join(cleaned.split())


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
