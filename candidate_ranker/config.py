"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for retrieval and OpenRouter model selection."""

    openrouter_primary_model: str
    top_k_retrieval: int
    top_k_features: int
    top_k_explain: int
    enable_ai_explanations: bool


def load_settings() -> Settings:
    """Load settings from `.env` and process environment."""

    load_dotenv()
    settings = Settings(
        openrouter_primary_model=os.getenv(
            "OPENROUTER_PRIMARY_MODEL", "deepseek/deepseek-chat-v3-0324"
        ),
        top_k_retrieval=int(os.getenv("APP_TOP_K_RETRIEVAL", "5000")),
        top_k_features=int(os.getenv("APP_TOP_K_FEATURES", "500")),
        top_k_explain=int(os.getenv("APP_TOP_K_EXPLAIN", "100")),
        enable_ai_explanations=os.getenv("APP_ENABLE_AI_EXPLANATIONS", "0").lower()
        in {"1", "true", "yes", "on"},
    )
    return settings
