# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the e2e harness' JSON/YAML boundaries.

Three things cross a serialization boundary and are parsed here (never by hand):

* :class:`E2EConfig`   -- ``config.yaml`` (what to serve + evaluate).
* :class:`Baseline`    -- a golden ``baselines/*.json`` the gate checks against.
* :class:`EvalResults` -- the ``results_*.json`` lm-eval/evalchemy writes.

Keeping these typed means a malformed config/baseline fails loudly at load with a
field-precise error, and the run/validate CLIs never spelunk raw dicts.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- config.yaml ---------------------------------------------------------------


class EvalSettings(BaseModel):
    """The evalchemy run: which tasks, how many samples, decoding."""

    model_config = ConfigDict(extra="forbid")

    tasks: List[str] = Field(default_factory=lambda: ["gsm8k"])
    apply_chat_template: bool = False
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    batch_size: Union[int, str] = 1
    seed: Optional[int] = 1234
    gen_kwargs: Optional[str] = None
    extra_model_args: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


class MarinServeSettings(BaseModel):
    """Defaults for the ``marin-serve`` provisioning provider."""

    model_config = ConfigDict(extra="forbid")

    cluster: str = "marin"
    tpu: str = "v5litepod-8"
    region: Optional[str] = None
    access: str = "link"
    wait_timeout_s: float = 1800.0
    timeout_hours: float = 2.0
    marin_workspace: Optional[str] = None


class ProviderSettings(BaseModel):
    """Per-provider config blocks (``provider:`` in the yaml)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    marin_serve: MarinServeSettings = Field(default_factory=MarinServeSettings, alias="marin-serve")
    endpoint: Dict[str, Any] = Field(default_factory=dict)


class E2EConfig(BaseModel):
    """The whole ``config.yaml``. Every field is overridable on the CLI."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    baseline: Optional[str] = None
    eval: EvalSettings = Field(default_factory=EvalSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)

    @classmethod
    def load(cls, path: str) -> "E2EConfig":
        import yaml  # local import: pure-JSON consumers (validate) need no yaml

        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    @classmethod
    def load_or_empty(cls, path: Optional[str]) -> "E2EConfig":
        if path and os.path.exists(path):
            return cls.load(path)
        return cls()


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


class Baseline(BaseModel):
    """A golden baseline: free-form provenance + per-task gate thresholds."""

    model_config = ConfigDict(extra="forbid")

    provenance: Dict[str, Any] = Field(default_factory=dict)
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
