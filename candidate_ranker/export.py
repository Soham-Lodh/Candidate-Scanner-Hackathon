"""CSV export helpers for ranked candidate results."""

from __future__ import annotations

import logging
import re
from io import StringIO

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

    return [
        {
            "candidate_id": score.candidate_id,
            "rank": index + 1,
            "score": score.composite_score,
            "reasoning": scoring_reasoning(score),
        }
        for index, score in enumerate(scores[:limit])
    ]


def scoring_reasoning(score: CandidateScore) -> str:
    """Explain ranking using local scoring evidence in recruiter-ready language."""

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
