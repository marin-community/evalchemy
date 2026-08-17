# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The run config for the serve-and-eval runner (e.g. ``configs/qwen-tiny.yaml``).

Describes what to serve and how to evaluate it. Layered by :meth:`RunConfig.load`:
CLI overrides win, then ``E2E_*`` env vars, then the yaml file, then these defaults.
The gate spec a run is checked against lives with the regression gate
(:mod:`eval.regression`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from evalchemy_config import EvaluationConfig

_EVALUATION_FIELDS = frozenset(EvaluationConfig.model_fields)


class RunConfig(BaseSettings):
    """What to serve and evaluate."""

    model_config = SettingsConfigDict(env_prefix="E2E_", extra="forbid", protected_namespaces=())

    _yaml_path: ClassVar[Optional[str]] = None

    # what to serve
    model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None

    # Evaluation intent is portable; the remaining fields are provider policy.
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # marin-serve provider
    cluster: str = "marin"
    tpu: str = "v6e-4"
    region: Optional[str] = None
    wait_timeout_s: float = 1800.0
    timeout_hours: float = 2.0
    marin_workspace: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_evaluation_fields(cls, value: Any) -> Any:
        """Accept existing flat runner YAML while storing evaluation intent once."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        evaluation = dict(data.get("evaluation", {}))
        for field in _EVALUATION_FIELDS:
            if field in data:
                evaluation.setdefault(field, data.pop(field))
        if evaluation:
            data["evaluation"] = evaluation
        return data

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        sources = [init_settings, env_settings]
        if cls._yaml_path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_path))
        return tuple(sources)

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "RunConfig":
        """Load from ``path`` (if it exists) + ``E2E_*`` env + CLI ``overrides``.

        ``None`` overrides are dropped so an unset CLI flag can't clobber the file.
        """
        cls._yaml_path = path if (path and os.path.exists(path)) else None
        evaluation = cls._evaluation_values_from_yaml(cls._yaml_path)
        evaluation.update(cls._evaluation_values_from_environment())
        init_values = {key: value for key, value in overrides.items() if value is not None}
        evaluation.update({key: init_values.pop(key) for key in list(init_values) if key in _EVALUATION_FIELDS})
        if evaluation:
            init_values["evaluation"] = evaluation
        return cls(**init_values)

    @staticmethod
    def _evaluation_values_from_yaml(path: Optional[str]) -> dict[str, Any]:
        if path is None:
            return {}
        with Path(path).open() as config_file:
            parsed = yaml.safe_load(config_file)
        if not isinstance(parsed, dict):
            return {}
        return {key: value for key, value in parsed.items() if key in _EVALUATION_FIELDS}

    @staticmethod
    def _evaluation_values_from_environment() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field in _EVALUATION_FIELDS:
            raw_value = os.environ.get(f"E2E_{field.upper()}")
            if raw_value is None:
                continue
            if field in {"tasks", "extra_model_args"}:
                values[field] = json.loads(raw_value)
            else:
                values[field] = raw_value
        return values

    @property
    def tasks(self) -> list[str]:
        return self.evaluation.tasks

    @property
    def apply_chat_template(self) -> bool:
        return self.evaluation.apply_chat_template

    @property
    def limit(self) -> int | None:
        return self.evaluation.limit

    @property
    def num_fewshot(self) -> int | None:
        return self.evaluation.num_fewshot

    @property
    def batch_size(self) -> int | str:
        return self.evaluation.batch_size

    @property
    def seed(self) -> int | None:
        return self.evaluation.seed

    @property
    def gen_kwargs(self) -> str | None:
        return self.evaluation.gen_kwargs

    @property
    def extra_model_args(self) -> dict[str, str | int | float | bool]:
        return self.evaluation.extra_model_args
