"""Provider adapters for reusable LLM judge calls."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Type

from .config import JudgeConfig, JudgeConfigurationError
from .schemas import JudgeRequest, JudgeResult


class JudgeError(RuntimeError):
    """Base class for judge-client failures."""


class JudgeProviderError(JudgeError):
    """Raised when a provider call fails."""


class JudgeResponseError(JudgeError):
    """Raised when a provider response cannot be parsed or validated."""


class JudgeClient:
    """Small provider-aware judge client.

    The OpenAI SDK is imported lazily unless tests inject ``openai_client_factory``.
    This keeps task discovery, config parsing, and non-judge runs free of provider
    imports and API-key checks.
    """

    def __init__(
        self,
        config: JudgeConfig,
        openai_client_factory: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self._openai_client_factory = openai_client_factory
        self._client = None

    def grade(self, request: JudgeRequest, schema: Type[JudgeResult] = JudgeResult) -> JudgeResult:
        if self.config.api_surface == "responses":
            return self._grade_responses(request, schema)
        if self.config.api_surface == "chat_completions":
            return self._grade_chat_completions(request, schema)
        raise JudgeConfigurationError(f"Unsupported judge API surface: {self.config.api_surface}")

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "api_key": self.config.validate_env(),
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return kwargs

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        factory = self._openai_client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise JudgeConfigurationError("The openai package is required for LLM judge calls.") from exc
            factory = OpenAI
        self._client = factory(**self._client_kwargs())
        return self._client

    def _grade_responses(self, request: JudgeRequest, schema: Type[JudgeResult]) -> JudgeResult:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": request.to_messages(schema.schema_name()),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.schema_name(),
                    "schema": schema.json_schema(),
                    "strict": True,
                }
            },
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        try:
            response = self._get_client().responses.create(**payload)
        except Exception as exc:  # pragma: no cover - provider-specific
            raise JudgeProviderError(f"{self.config.provider} judge call failed: {exc}") from exc

        return self._parse_response_text(_extract_responses_text(response), schema, response)

    def _grade_chat_completions(self, request: JudgeRequest, schema: Type[JudgeResult]) -> JudgeResult:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": request.to_messages(schema.schema_name()),
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.provider == "deepseek":
            payload["response_format"] = {"type": "json_object"}
            if self.config.reasoning_effort:
                payload["reasoning_effort"] = self.config.reasoning_effort
                payload["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.schema_name(),
                    "schema": schema.json_schema(),
                    "strict": True,
                },
            }
            if self.config.temperature is not None:
                payload["temperature"] = self.config.temperature

        try:
            response = self._get_client().chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - provider-specific
            raise JudgeProviderError(f"{self.config.provider} judge call failed: {exc}") from exc

        return self._parse_response_text(_extract_chat_text(response), schema, response)

    def _parse_response_text(
        self,
        text: str,
        schema: Type[JudgeResult],
        provider_response: Optional[Any] = None,
    ) -> JudgeResult:
        try:
            payload = _loads_json_object(text)
            result = schema.from_dict(payload)
        except Exception as exc:
            raise JudgeResponseError(f"Judge response did not match {schema.schema_name()}: {exc}") from exc

        result.judge_model = self.config.model
        result.judge_provider = self.config.provider
        result.judge_config_hash = self.config.config_hash()
        result.usage = _extract_usage(provider_response)
        result.provider_response_id = _extract_response_id(provider_response)
        return result


def _loads_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response content")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("top-level JSON value is not an object")
    return loaded


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return getattr(message, "content", "") or ""


def _extract_usage(response: Any) -> Optional[Dict[str, Any]]:
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(getattr(usage, "__dict__", {}) or {})


def _extract_response_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    response_id = getattr(response, "id", None)
    return str(response_id) if response_id is not None else None
