"""Validation tests for dynamic schema mapping."""

from candidate_ranker.schema_mapping import build_schema_map, resolve_paths


def test_schema_map_uses_dynamic_paths() -> None:
    schema = {
        "type": "object",
        "properties": {
            "profile": {"type": "object", "properties": {"full_name": {"type": "string"}}},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "work_history": {
                "type": "array",
                "items": {"type": "object", "properties": {"years": {"type": "number"}}},
            },
        },
    }
    schema_map = build_schema_map(schema)
    assert "$.capabilities" in schema_map.skills.paths
    assert "$.work_history" in schema_map.experience.paths
    candidate = {"profile": {"full_name": "Ada"}, "capabilities": ["Python"]}
    assert resolve_paths(candidate, schema_map.identity.paths)[0] == "Ada"
