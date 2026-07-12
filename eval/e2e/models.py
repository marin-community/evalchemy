# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the harness's JSON/YAML files:

* :class:`E2EConfig`   -- the run config (e.g. ``qwen-tiny.yaml``).
* :class:`Baseline`    -- a golden ``baselines/*.json`` the gate checks against.
* :class:`EvalResults` -- the ``results_*.json`` lm-eval/evalchemy writes.

A malformed config or baseline fails at load with a field-precise error.
"""

from __future__ import annotations

import glob
import os
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

# --- the run config (e.g. qwen-tiny.yaml) --------------------------------------


class E2EConfig(BaseSettings):
    """What to serve and evaluate.

    Layered by :meth:`load`: CLI overrides win, then ``E2E_*`` env vars, then the
    yaml file (e.g. ``qwen-tiny.yaml``), then these defaults.
    """

    model_config = SettingsConfigDict(env_prefix="E2E_", extra="forbid", protected_namespaces=())

    _yaml_path: ClassVar[Optional[str]] = None

    # what to serve
    model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    baseline: Optional[str] = None

    # the evalchemy run
    tasks: List[str] = Field(default_factory=lambda: ["gsm8k"])
    apply_chat_template: bool = False
    limit: Optional[int] = 200
    num_fewshot: Optional[int] = None
    batch_size: Union[int, str] = 1
    seed: Optional[int] = 1234
    gen_kwargs: Optional[str] = None
    extra_model_args: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    # marin-serve provider
    cluster: str = "marin"
    tpu: str = "v5litepod-8"
    region: Optional[str] = None
    access: str = "link"
    wait_timeout_s: float = 1800.0
    timeout_hours: float = 2.0
    marin_workspace: Optional[str] = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        sources = [init_settings, env_settings]
        if cls._yaml_path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_path))
        return tuple(sources)

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "E2EConfig":
        """Load from ``path`` (if it exists) + ``E2E_*`` env + CLI ``overrides``.

        ``None`` overrides are dropped so an unset CLI flag can't clobber the file.
        """
        cls._yaml_path = path if (path and os.path.exists(path)) else None
        return cls(**{k: v for k, v in overrides.items() if v is not None})


# --- baselines/*.json ----------------------------------------------------------


class MetricThreshold(BaseModel):
    """Gate thresholds for one metric.

    ``min`` is the coarse "not broken/empty" floor. ``reference``/``tolerance`` are
    an OPTIONAL tighter band (``|observed - reference| <= tolerance``); a reference
    without a tolerance is meaningless, so it is rejected at parse time.
    """

    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    reference: Optional[float] = None
    tolerance: Optional[float] = None

    @model_validator(mode="after")
    def _reference_requires_tolerance(self) -> "MetricThreshold":
        if self.reference is not None and self.tolerance is None:
            raise ValueError("'reference' requires 'tolerance'")
        return self


class TaskBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: Dict[str, MetricThreshold] = Field(default_factory=dict)
    observed: Dict[str, float] = Field(default_factory=dict)
    expected_samples: Optional[int] = None


class BaselineProvenance(BaseModel):
    """How a baseline was produced. Informational -- the gate never reads it."""

    # extra="ignore": tolerate fields older/newer recorders wrote. protected_namespaces
    # lets `model` / `model_revision` coexist with pydantic's `model_` namespace.
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    lm_eval_version: Optional[str] = None
    adapter: Optional[str] = None
    apply_chat_template: Optional[bool] = None
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    seed: Optional[int] = None
    recorded_at: Optional[str] = None
    note: Optional[str] = None


class Baseline(BaseModel):
    """A golden baseline: provenance + per-task gate thresholds."""

    model_config = ConfigDict(extra="forbid")

    provenance: BaselineProvenance = Field(default_factory=BaselineProvenance)
    tasks: Dict[str, TaskBaseline]

    @classmethod
    def load(cls, path: str) -> "Baseline":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, exclude_none=True))
            f.write("\n")


# --- results_*.json (lm-eval / evalchemy output) -------------------------------


class EvalResults(BaseModel):
    """The subset of an lm-eval ``results_*.json`` the harness consumes.

    lm-eval writes a large document; ``extra="ignore"`` keeps only what we read.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    n_samples: Dict[str, Any] = Field(default_factory=dict, alias="n-samples")
    lm_eval_version: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    model_source: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "EvalResults":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @staticmethod
    def find_latest_path(output_dir: str) -> str:
        """Newest ``results_*.json`` under ``output_dir`` (searched recursively).

        evalchemy writes ``<output_dir>/<model_sanitized>/results_<ts>.json``.
        """
        matches = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
        matches += glob.glob(os.path.join(output_dir, "results_*.json"))
        if not matches:
            raise FileNotFoundError(f"no results_*.json found under {output_dir!r}")
        return max(set(matches), key=os.path.getmtime)

    @classmethod
    def load_path_or_dir(cls, path: str) -> "EvalResults":
        """Load a results file, or find the latest one if ``path`` is a directory."""
        if os.path.isdir(path):
            path = cls.find_latest_path(path)
        return cls.load(path)

    def metric(self, task: str, name: str) -> Optional[float]:
        value = (self.results.get(task) or {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    def numeric_metrics(self, task: str) -> Dict[str, float]:
        task_results = self.results.get(task) or {}
        return {k: float(v) for k, v in task_results.items() if isinstance(v, (int, float))}

    def sample_count(self, task: str) -> Optional[int]:
        """Best-effort effective sample count for ``task``.

        lm-eval's ``--limit`` paths populate top-level ``n-samples``; evalchemy's
        lm-eval-native (gsm8k) path instead records ``sample_len`` on the task.
        """
        entry = self.n_samples.get(task)
        if isinstance(entry, dict):
            for key in ("effective", "original"):
                if isinstance(entry.get(key), int):
                    return int(entry[key])
        sample_len = (self.results.get(task) or {}).get("sample_len")
        return int(sample_len) if isinstance(sample_len, int) else None
