"""Configuration for reusable LLM judge providers.

The runner should carry provider routing, model IDs, and non-secret knobs through
metadata and resume fingerprints, but it should never materialize or persist API
keys. Judge calls resolve the configured environment variable only at call time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional


class JudgeConfigurationError(ValueError):
    """Raised when judge configuration is invalid or missing credentials."""


JUDGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "gpt-5.5": {
        "provider": "openai",
        "api_surface": "responses",
        "api_key_env": "OPENAI_API_KEY",
        # OpenAI's GPT-5.5 docs recommend medium as the balanced starting point.
        "reasoning_effort": "medium",
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "api_surface": "chat_completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        # DeepSeek's thinking-mode docs default regular thinking requests to high.
        "reasoning_effort": "high",
    },
}


_ARG_TO_FIELD = {
    "annotator_provider": "provider",
    "annotator_api_surface": "api_surface",
    "annotator_api_key_env": "api_key_env",
    "annotator_base_url": "base_url",
    "annotator_reasoning_effort": "reasoning_effort",
    "annotator_max_output_tokens": "max_output_tokens",
    "annotator_num_workers": "num_workers",
    "annotator_cache_path": "cache_path",
}

_YAML_TO_ARG = {
    "provider": "annotator_provider",
    "api_surface": "annotator_api_surface",
    "api_key_env": "annotator_api_key_env",
    "base_url": "annotator_base_url",
    "reasoning_effort": "annotator_reasoning_effort",
    "max_output_tokens": "annotator_max_output_tokens",
    "num_workers": "annotator_num_workers",
    "cache_path": "annotator_cache_path",
}


def add_judge_config_args(parser: Any) -> None:
    """Register reusable judge-provider flags on an argparse parser."""

    annotator_group = parser.add_argument_group("annotator")
    annotator_group.add_argument(
        "--annotator_provider",
        type=str,
        default=None,
        choices=["openai", "deepseek"],
        help="Provider for reusable LLM judge tasks. Inferred from --annotator_model when omitted.",
    )
    annotator_group.add_argument(
        "--annotator_api_surface",
        type=str,
        default=None,
        choices=["responses", "chat_completions"],
        help="Provider API surface for reusable LLM judge tasks.",
    )
    annotator_group.add_argument(
        "--annotator_api_key_env",
        type=str,
        default=None,
        help="Environment variable containing the judge provider API key.",
    )
    annotator_group.add_argument(
        "--annotator_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for the judge provider.",
    )
    annotator_group.add_argument(
        "--annotator_reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort for judge providers that support it.",
    )
    annotator_group.add_argument(
        "--annotator_max_output_tokens",
        type=int,
        default=None,
        help="Maximum output tokens for LLM judge calls.",
    )
    annotator_group.add_argument(
        "--annotator_num_workers",
        type=int,
        default=None,
        help="Maximum parallel LLM judge workers for judge-capable benchmarks.",
    )
    annotator_group.add_argument(
        "--annotator_cache_path",
        type=str,
        default=None,
        help="Optional cache path for judge-capable benchmarks.",
    )


def apply_annotator_config_overrides(args: Any, raw_yaml: Mapping[str, Any]) -> None:
    """Apply the annotator portion of Evalchemy YAML config to parsed args."""

    annotator = raw_yaml.get("annotator")
    if annotator is not None:
        if not isinstance(annotator, Mapping):
            raise JudgeConfigurationError("YAML field 'annotator' must be a mapping.")
        args.annotator_model = annotator.get("model", args.annotator_model)
        for yaml_key, arg_name in _YAML_TO_ARG.items():
            if yaml_key in annotator:
                setattr(args, arg_name, annotator[yaml_key])
    else:
        args.annotator_model = raw_yaml.get("annotator_model", args.annotator_model)


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved, non-secret LLM judge configuration."""

    provider: str = "openai"
    model: str = "gpt-5.5"
    api_surface: str = "responses"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = "medium"
    temperature: Optional[float] = None
    timeout: float = 300.0
    max_retries: int = 2
    max_output_tokens: int = 4096
    num_workers: int = 2
    cache_path: Optional[str] = None

    @classmethod
    def from_model(cls, model_name: Optional[str], **overrides: Any) -> "JudgeConfig":
        """Resolve a model name plus explicit overrides into a judge config."""

        provider_override = overrides.get("provider")
        model = model_name or "gpt-5.5"
        if model == "auto":
            model = "deepseek-v4-pro" if provider_override == "deepseek" else "gpt-5.5"

        values: Dict[str, Any] = {"model": model}
        values.update(_infer_defaults(model, provider_override))
        values.update(JUDGE_PRESETS.get(model, {}))

        for key, value in overrides.items():
            if value is None or value == "":
                continue
            values[key] = value

        config = cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})
        config.validate_shape()
        return config

    @classmethod
    def from_args(cls, args: Any) -> Optional["JudgeConfig"]:
        """Build judge config from parsed CLI args.

        Existing Evalchemy runs default ``--annotator_model`` to ``auto``. Returning
        ``None`` for the untouched default preserves non-judge behavior and avoids
        adding irrelevant judge settings to metadata or resume fingerprints.
        """

        model = getattr(args, "annotator_model", None)
        overrides = _overrides_from_args(args)
        if not _should_build_config(model, overrides):
            return None
        return cls.from_model(model, **overrides)

    @classmethod
    def from_yaml(cls, raw_yaml: Mapping[str, Any], args_defaults: Optional[Any] = None) -> Optional["JudgeConfig"]:
        """Build judge config from Evalchemy YAML config data."""

        annotator = raw_yaml.get("annotator") or {}
        if annotator and not isinstance(annotator, Mapping):
            raise JudgeConfigurationError("annotator config must be a mapping")

        model = annotator.get("model") if annotator else raw_yaml.get("annotator_model")
        overrides = {
            "provider": annotator.get("provider"),
            "api_surface": annotator.get("api_surface"),
            "api_key_env": annotator.get("api_key_env"),
            "base_url": annotator.get("base_url"),
            "reasoning_effort": annotator.get("reasoning_effort"),
            "temperature": annotator.get("temperature"),
            "timeout": annotator.get("timeout"),
            "max_retries": annotator.get("max_retries"),
            "max_output_tokens": annotator.get("max_output_tokens"),
            "num_workers": annotator.get("num_workers"),
            "cache_path": annotator.get("cache_path"),
        }

        if args_defaults is not None:
            args_config = cls.from_args(args_defaults)
            if args_config is not None:
                base = asdict(args_config)
                for key, value in overrides.items():
                    if value is not None and value != "":
                        base[key] = value
                if model is not None:
                    base["model"] = model
                return cls.from_model(base.pop("model"), **base)

        clean_overrides = {key: value for key, value in overrides.items() if value is not None and value != ""}
        if not _should_build_config(model, clean_overrides):
            return None
        return cls.from_model(model, **clean_overrides)

    def with_overrides(self, **overrides: Any) -> "JudgeConfig":
        clean = {key: value for key, value in overrides.items() if value is not None and value != ""}
        config = replace(self, **clean)
        config.validate_shape()
        return config

    def validate_shape(self) -> None:
        if self.provider not in {"openai", "deepseek"}:
            raise JudgeConfigurationError(f"Unsupported judge provider: {self.provider}")
        if self.api_surface not in {"responses", "chat_completions"}:
            raise JudgeConfigurationError(f"Unsupported judge API surface: {self.api_surface}")
        if self.provider == "deepseek" and self.api_surface != "chat_completions":
            raise JudgeConfigurationError("DeepSeek judge provider requires chat_completions")
        if self.provider == "deepseek" and not self.base_url:
            raise JudgeConfigurationError("DeepSeek judge provider requires base_url")
        if not self.model:
            raise JudgeConfigurationError("Judge model must be set")
        if not self.api_key_env:
            raise JudgeConfigurationError("Judge api_key_env must be set")
        if self.max_output_tokens <= 0:
            raise JudgeConfigurationError("Judge max_output_tokens must be positive")
        if self.num_workers <= 0:
            raise JudgeConfigurationError("Judge num_workers must be positive")

    def validate_env(self) -> str:
        """Return the configured API key, or raise a clear missing-env error."""

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise JudgeConfigurationError(
                f"Please set {self.api_key_env} to use {self.model} as an LLM judge."
            )
        return api_key

    def redacted_dict(self) -> Dict[str, Any]:
        """Return stable non-secret config data for logs and fingerprints."""

        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    def config_hash(self) -> str:
        payload = json.dumps(self.redacted_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _overrides_from_args(args: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg_name, field_name in _ARG_TO_FIELD.items():
        value = getattr(args, arg_name, None)
        if value is not None and value != "":
            overrides[field_name] = value
    return overrides


def _infer_defaults(model: str, provider_override: Optional[str]) -> Dict[str, Any]:
    if provider_override == "deepseek" or model.startswith("deepseek-"):
        return {
            "provider": "deepseek",
            "api_surface": "chat_completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "reasoning_effort": "high",
        }
    return {
        "provider": "openai",
        "api_surface": "responses",
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_effort": "medium",
    }


def _should_build_config(model: Optional[str], overrides: Mapping[str, Any]) -> bool:
    """Return true when inputs opt into the reusable judge layer.

    Legacy Evalchemy judge benchmarks already use ``annotator_model`` directly.
    Creating a shared ``JudgeConfig`` for every non-auto legacy model would change
    metadata and resume fingerprints without changing benchmark behavior. Presets
    and explicit provider knobs opt into the new layer; old flat annotator models
    remain untouched.
    """

    clean_overrides = {key: value for key, value in overrides.items() if value is not None and value != ""}
    if clean_overrides:
        return True
    if model is None or model == "auto":
        return False
    return model in JUDGE_PRESETS or str(model).startswith("deepseek-")
