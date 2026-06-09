"""OpenRouter-only asynchronous client with retries, JSON validation, and model failover.

Example:
    client = OpenRouterClient()
    payload = await client.chat_json(messages=[{"role": "user", "content": "Return JSON"}])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen3-235b-a22b",
    "meta-llama/llama-3.3-70b-instruct",
]
BACKOFF_SECONDS = [2, 4, 8]


class OpenRouterError(RuntimeError):
    """Base exception for OpenRouter client failures."""


class OpenRouterAuthError(OpenRouterError):
    """Raised when API authentication is missing or rejected."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when a model or account is rate limited."""


class OpenRouterInvalidJSONError(OpenRouterError):
    """Raised when a response cannot be parsed as valid JSON."""


class OpenRouterValidationError(OpenRouterError):
    """Raised when parsed JSON fails Pydantic validation."""


@dataclass(slots=True)
class OpenRouterClient:
    """HTTP client wrapper that enforces all LLM access through OpenRouter."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    timeout_seconds: float = 45.0
    client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", self.base_url)
        timeout_env = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
        if timeout_env:
            self.timeout_seconds = float(timeout_env)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        deterministic: bool = False,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return raw OpenRouter chat completion JSON with automatic model failover."""

        if not self.api_key:
            raise OpenRouterAuthError("OPENROUTER_API_KEY is required for AI operations.")

        model_order = self._model_order(model)
        last_error: Exception | None = None
        for selected_model in model_order:
            try:
                return await self.retry_with_backoff(
                    self._post_completion,
                    messages=messages,
                    model=selected_model,
                    temperature=0.0 if deterministic else temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except OpenRouterError as exc:
                LOGGER.warning("Model %s failed after retries: %s", selected_model, exc)
                last_error = exc
        raise OpenRouterError(f"All OpenRouter models failed: {last_error}") from last_error

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        deterministic: bool = False,
    ) -> BaseModel:
        """Return a Pydantic-validated JSON object from a chat completion."""

        raw = await self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            deterministic=deterministic,
            response_format={"type": "json_object"},
        )
        content = self._extract_content(raw)
        parsed = self.validate_response(content, schema)
        LOGGER.info("Validated OpenRouter JSON response as %s", schema.__name__)
        return parsed

    def validate_response(self, content: str, schema: type[BaseModel]) -> BaseModel:
        """Parse and validate strict JSON content using Pydantic."""

        try:
            data = json.loads(self._strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise OpenRouterInvalidJSONError(f"Model returned invalid JSON: {exc}") from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise OpenRouterValidationError(str(exc)) from exc

    async def retry_with_backoff(self, func: Any, **kwargs: Any) -> dict[str, Any]:
        """Retry transient failures with exponential backoff before failing over models."""

        last_error: Exception | None = None
        for attempt, delay in enumerate([0, *BACKOFF_SECONDS], start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                return await func(**kwargs)
            except (OpenRouterRateLimitError, OpenRouterInvalidJSONError, httpx.TimeoutException, httpx.HTTPError, OpenRouterError) as exc:
                LOGGER.warning("OpenRouter attempt %s failed: %s", attempt, exc)
                last_error = exc
        raise OpenRouterError(f"Retries exhausted: {last_error}") from last_error

    async def _post_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        LOGGER.info(
            "OpenRouter request model=%s temperature=%s max_tokens=%s messages=%s",
            model,
            temperature,
            max_tokens,
            len(messages),
        )
        close_client = self.client is None
        async_client = self.client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await async_client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://local.candidate-ranker",
                    "X-Title": "AI Candidate Ranking Platform",
                },
                json=payload,
            )
            if response.status_code == 401:
                raise OpenRouterAuthError("OpenRouter authentication failed.")
            if response.status_code == 429:
                raise OpenRouterRateLimitError("OpenRouter rate limit reached.")
            if response.status_code >= 400:
                raise OpenRouterError(f"OpenRouter HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            LOGGER.info("OpenRouter response received model=%s", model)
            return data
        except json.JSONDecodeError as exc:
            raise OpenRouterInvalidJSONError("OpenRouter returned non-JSON HTTP body.") from exc
        finally:
            if close_client:
                await async_client.aclose()

    def _model_order(self, requested: str | None) -> list[str]:
        configured = list(dict.fromkeys([requested or "", *self.models]))
        return [model for model in configured if model]

    @staticmethod
    def _extract_content(raw: dict[str, Any]) -> str:
        try:
            return str(raw["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("Unexpected OpenRouter response shape.") from exc

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped
