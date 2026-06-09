"""Text normalization and lightweight NLP helpers for local ranking."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}")


def normalize_text(value: Any) -> str:
    """Convert nested candidate values into searchable normalized text."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(normalize_text(v) for v in value.values())
    if isinstance(value, list | tuple | set):
        return " ".join(normalize_text(v) for v in value)
    return re.sub(r"\s+", " ", str(value)).strip()


def tokens(text: str) -> set[str]:
    """Extract lowercase searchable tokens."""

    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def skill_overlap(candidate_text: str, desired: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return matched and missing skills using normalized substring and token matching."""

    lowered = candidate_text.lower()
    candidate_tokens = tokens(candidate_text)
    matched: list[str] = []
    missing: list[str] = []
    for skill in desired:
        normalized = skill.strip().lower()
        if not normalized:
            continue
        skill_parts = tokens(normalized)
        if normalized in lowered or (skill_parts and skill_parts.issubset(candidate_tokens)):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing
