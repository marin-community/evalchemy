# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Read Evalchemy's aggregate results from its FineStore archive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.contracts.finestore_output import read_finestore_output
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
    def load_archive(cls, root: str, source_prefix: str = "run") -> "EvalResults":
        """Load one task group's committed aggregate from FineStore."""
        document = read_finestore_output(root, source_prefix)
        if document is None:
            raise FileNotFoundError(f"no Evalchemy result for {source_prefix!r} in FineStore archive {root!r}")
        return cls.model_validate(document)

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
