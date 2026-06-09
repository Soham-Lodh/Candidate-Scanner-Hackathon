"""Dynamic JSON Schema analysis and field-path resolution."""

from __future__ import annotations

import logging
from typing import Any

from candidate_ranker.models import FieldMapping, SchemaMap

LOGGER = logging.getLogger(__name__)

SEMANTIC_KEYWORDS: dict[str, list[str]] = {
    "identity": ["name", "full_name", "candidate", "email", "id", "profile"],
    "skills": ["skill", "technology", "tool", "stack", "competency", "capabilit", "keyword"],
    "experience": ["experience", "employment", "work", "history", "years", "role"],
    "education": ["education", "degree", "university", "school", "certification"],
    "location": ["location", "city", "country", "state", "remote", "timezone"],
    "availability": ["availability", "notice", "start", "joining"],
    "seniority": ["seniority", "level", "grade", "title"],
    "compensation": ["salary", "compensation", "rate", "ctc"],
    "projects": ["project", "portfolio", "achievement", "production", "impact"],
}


def build_schema_map(schema: dict[str, Any]) -> SchemaMap:
    """Build a semantic map from JSON Schema Draft 7 without hardcoded candidate fields."""

    paths = list(iter_schema_paths(schema))
    mappings = {
        name: FieldMapping(name=name, paths=_rank_paths(paths, keywords), confidence=0.0)
        for name, keywords in SEMANTIC_KEYWORDS.items()
    }
    for mapping in mappings.values():
        mapping.confidence = min(1.0, len(mapping.paths) / 3) if mapping.paths else 0.0
    LOGGER.info("Schema map created with %s paths", len(paths))
    return SchemaMap(raw_paths=paths, **mappings)


def iter_schema_paths(schema: dict[str, Any], prefix: str = "$") -> list[str]:
    """Return candidate data paths from schema properties and array items."""

    discovered: list[str] = []
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    for key, subschema in properties.items():
        path = f"{prefix}.{key}"
        discovered.append(path)
        if isinstance(subschema, dict):
            if subschema.get("type") == "object" or "properties" in subschema:
                discovered.extend(iter_schema_paths(subschema, path))
            items = subschema.get("items")
            if isinstance(items, dict):
                item_path = f"{path}[]"
                discovered.append(item_path)
                discovered.extend(iter_schema_paths(items, item_path))
    return discovered


def resolve_paths(candidate: dict[str, Any], paths: list[str]) -> list[Any]:
    """Resolve mapped JSON-style paths against a candidate record."""

    values: list[Any] = []
    for path in paths:
        values.extend(_resolve_one(candidate, path.replace("$.", "").split(".")))
    return [value for value in values if value not in (None, "", [], {})]


def display_name(candidate: dict[str, Any], schema_map: SchemaMap) -> str:
    """Derive a display name through dynamic identity paths."""

    preferred_paths = _display_name_paths(schema_map.identity.paths)
    for value in resolve_paths(candidate, preferred_paths):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float):
            return str(value)
    return str(
        candidate.get("candidate_id")
        or candidate.get("id")
        or candidate.get("_row_id")
        or "Unknown Candidate"
    )


def _display_name_paths(paths: list[str]) -> list[str]:
    name_paths = [path for path in paths if "name" in path.lower()]
    non_id_paths = [
        path
        for path in paths
        if path not in name_paths and not path.lower().endswith("_id") and ".id" not in path.lower()
    ]
    id_paths = [path for path in paths if path not in name_paths and path not in non_id_paths]
    return [*name_paths, *non_id_paths, *id_paths]


def _rank_paths(paths: list[str], keywords: list[str]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for path in paths:
        lowered = path.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [path for _, path in scored[:8]]


def _resolve_one(current: Any, parts: list[str]) -> list[Any]:
    if not parts:
        return [current]
    part = parts[0]
    is_array = part.endswith("[]")
    key = part[:-2] if is_array else part
    if isinstance(current, dict):
        next_value = current.get(key)
    else:
        return []
    if is_array:
        if isinstance(next_value, list):
            values: list[Any] = []
            for item in next_value:
                values.extend(_resolve_one(item, parts[1:]))
            return values
        return []
    return _resolve_one(next_value, parts[1:])
