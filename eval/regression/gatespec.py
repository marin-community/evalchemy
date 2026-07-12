# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for a regression gate spec (``specs/*.json``).

A :class:`GateSpec` is a coarse smoke-threshold set: per-metric floors plus an expected
sample count (see :mod:`eval.regression.gate`). A malformed spec fails at load with a
field-precise error.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricThreshold(BaseModel):
    """Gate thresholds for one metric.

    ``min`` is the coarse "not broken/empty" floor. ``reference``/``tolerance`` are an
    OPTIONAL tighter two-sided band (``|observed - reference| <= tolerance``) for a
    higher-limit regression gate; a reference without a tolerance is meaningless, so
    it is rejected at parse time.
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


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: Dict[str, MetricThreshold] = Field(default_factory=dict)
    observed: Dict[str, float] = Field(default_factory=dict)
    expected_samples: Optional[int] = None


class SpecProvenance(BaseModel):
    """How a spec was produced -- informational context for whoever reads the spec."""

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


class GateSpec(BaseModel):
    """A regression gate spec: provenance + per-task thresholds."""

    model_config = ConfigDict(extra="forbid")

    provenance: SpecProvenance = Field(default_factory=SpecProvenance)
    tasks: Dict[str, TaskSpec]

    @classmethod
    def load(cls, path: str) -> "GateSpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, exclude_none=True))
            f.write("\n")
