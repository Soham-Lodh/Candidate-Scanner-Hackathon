"""Typed domain models used across ingestion, schema mapping, ranking, and AI calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MODEL_OPTIONS: dict[str, str] = {
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324",
    "Qwen 3 235B": "qwen/qwen3-235b-a22b",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
}


class FieldMapping(BaseModel):
    """A semantic candidate attribute mapped to zero or more JSON paths."""

    name: str
    paths: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SchemaMap(BaseModel):
    """Dynamic field map derived from an uploaded JSON schema."""

    identity: FieldMapping
    skills: FieldMapping
    experience: FieldMapping
    education: FieldMapping
    location: FieldMapping
    availability: FieldMapping
    seniority: FieldMapping
    compensation: FieldMapping
    projects: FieldMapping
    raw_paths: list[str] = Field(default_factory=list)


class JDIntelligence(BaseModel):
    """Structured intelligence extracted from a job description and schema."""

    title: str = ""
    seniority: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    education_keywords: list[str] = Field(default_factory=list)
    experience_years_min: float | None = None
    production_signals: list[str] = Field(default_factory=list)
    behavioral_signals: list[str] = Field(default_factory=list)
    schema_notes: list[str] = Field(default_factory=list)


class CandidateScore(BaseModel):
    """Explainable local score for a candidate."""

    candidate_id: str
    display_name: str
    composite_score: float
    breakdown: dict[str, float]
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    concerns: list[str]
    integrity_flags: list[str]
    raw_candidate: dict[str, Any]


class CandidateExplanation(BaseModel):
    """AI-generated explanation for a ranked candidate."""

    candidate_id: str
    summary: str
    recruiter_rationale: list[str]
    interview_focus: list[str]
    risk_notes: list[str]


class ExplanationBatch(BaseModel):
    """Strict JSON wrapper for top-candidate explanations."""

    explanations: list[CandidateExplanation] = Field(default_factory=list)
