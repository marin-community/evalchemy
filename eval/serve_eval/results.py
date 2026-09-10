# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Read the ``results_*.json`` that lm-eval / evalchemy writes.

A read model shared by the runner (to summarize a run) and the regression gate (to
compare a run against a spec); it models evalchemy's output schema. lm-eval writes a
large document; ``extra="ignore"`` keeps only the subset we read.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.contracts.task_outcome import TaskOutcome, lm_eval_task_counts, validate_result_document


class EvalResults(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    task_outcomes: Dict[str, TaskOutcome] = Field(default_factory=dict)
    task_preparations: Dict[str, Any] = Field(default_factory=dict)
    generation_artifacts: Dict[str, Any] = Field(default_factory=dict)
    n_samples: Dict[str, Any] = Field(default_factory=dict, alias="n-samples")
    lm_eval_version: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    model_source: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def require_shared_result_contract(cls, value: Any) -> Any:
        """Reject legacy or malformed result dictionaries at every read path."""
        if isinstance(value, Mapping):
            validate_result_document(value)
        return value

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
        """Return lm-eval's effective sample count when the result supplies one."""
        _, effective = lm_eval_task_counts(
            task,
            {"n-samples": self.n_samples, "results": self.results},
        )
        return effective
