"""Unit tests for the OpenRouter client wrapper."""

import pytest
from pydantic import BaseModel

from ai.openrouter_client import OpenRouterClient, OpenRouterInvalidJSONError


class TinySchema(BaseModel):
    """Small schema for JSON validation tests."""

    name: str


def test_validate_response_accepts_strict_json() -> None:
    client = OpenRouterClient(api_key="test")
    parsed = client.validate_response('{"name":"Ada"}', TinySchema)
    assert parsed.name == "Ada"


def test_validate_response_rejects_invalid_json() -> None:
    client = OpenRouterClient(api_key="test")
    with pytest.raises(OpenRouterInvalidJSONError):
        client.validate_response("not-json", TinySchema)


def test_requested_model_is_first_in_failover_order() -> None:
    client = OpenRouterClient(api_key="test")
    order = client._model_order("qwen/qwen3-235b-a22b:free")
    assert order[0] == "qwen/qwen3-235b-a22b:free"
