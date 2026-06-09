"""High-level orchestration service for the ranking pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from candidate_ranker.ai_service import extract_jd_intelligence, generate_candidate_explanations
from candidate_ranker.config import Settings
from candidate_ranker.ingestion import iter_jsonl_candidates
from candidate_ranker.models import CandidateScore, ExplanationBatch, JDIntelligence, SchemaMap
from candidate_ranker.ranking import rank_candidates, rank_candidates_stream
from candidate_ranker.retrieval import SemanticRetriever
from candidate_ranker.schema_mapping import build_schema_map

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RankingResult:
    """Full result bundle for UI rendering and export."""

    schema_map: SchemaMap
    jd_intelligence: JDIntelligence
    ranked: list[CandidateScore]
    explanations: ExplanationBatch


async def run_pipeline(
    *,
    jd_text: str,
    candidates: list[dict[str, Any]],
    schema: dict[str, Any],
    settings: Settings,
    model: str,
) -> RankingResult:
    """Run schema mapping, two AI calls, retrieval funnel, scoring, and explanations."""

    schema_map = build_schema_map(schema)
    jd_intelligence = await extract_jd_intelligence(jd_text, schema_map, model=model)
    retrieved = SemanticRetriever().retrieve(jd_text, candidates, settings.top_k_retrieval)
    feature_ranked = rank_candidates(retrieved, jd_intelligence, schema_map, settings.top_k_features)
    integrity_slice = feature_ranked[: min(settings.top_k_explain, len(feature_ranked))]
    explanations = (
        await generate_candidate_explanations(jd_intelligence, integrity_slice, model=model)
        if settings.enable_ai_explanations
        else ExplanationBatch()
    )
    LOGGER.info("Pipeline completed with %s final ranked candidates", len(feature_ranked))
    return RankingResult(schema_map, jd_intelligence, feature_ranked, explanations)


async def run_pipeline_from_jsonl(
    *,
    jd_text: str,
    candidate_path: str,
    schema: dict[str, Any],
    settings: Settings,
    model: str,
) -> RankingResult:
    """Run ranking from a disk-backed JSONL candidate file."""

    schema_map = build_schema_map(schema)
    jd_intelligence = await extract_jd_intelligence(jd_text, schema_map, model=model)
    feature_ranked = rank_candidates_stream(
        iter_jsonl_candidates(candidate_path),
        jd_intelligence,
        schema_map,
        settings.top_k_features,
    )
    integrity_slice = feature_ranked[: min(settings.top_k_explain, len(feature_ranked))]
    explanations = (
        await generate_candidate_explanations(jd_intelligence, integrity_slice, model=model)
        if settings.enable_ai_explanations
        else ExplanationBatch()
    )
    LOGGER.info("Disk-backed pipeline completed with %s final ranked candidates", len(feature_ranked))
    return RankingResult(schema_map, jd_intelligence, feature_ranked, explanations)
