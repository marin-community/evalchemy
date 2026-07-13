# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check an eval run against a regression gate spec, or record a new one.

``check`` compares a run's ``results_*.json`` to a spec and exits non-zero on
regression; ``record`` writes a spec from a run. The gate is a coarse smoke check: a
small ``--limit`` run of a 0.6B model moves in coarse steps (gsm8k strict-match swings
~3/20 run-to-run even greedy), so it asserts the endpoint answered the expected sample
count and each metric clears a floor -- plus, optionally, stays within ``tolerance`` of
a recorded ``reference``. See ``eval/regression/README.md``.

    python -m eval.regression.validate check  --results <run-dir> --spec eval/regression/specs/qwen3-0.6b.json
    python -m eval.regression.validate record --results <run-dir> --spec eval/regression/specs/qwen3-0.6b.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import click
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.serve_eval.results import EvalResults

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SPEC = os.path.join(_REPO_ROOT, "eval", "regression", "specs", "qwen3-0.6b.json")


# --- the gate spec (specs/*.json) ---------------------------------------------


class MetricThreshold(BaseModel):
    """Gate thresholds for one metric.

    ``min`` is the coarse "not broken/empty" floor. ``reference``/``tolerance`` are an
    optional tighter two-sided band (``|observed - reference| <= tolerance``) for a
    higher-limit gate; a reference without a tolerance is meaningless, so it is
    rejected at parse time.
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
    """How a spec was produced -- context for whoever reads the spec."""

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


# --- the gate decision --------------------------------------------------------


@dataclass
class MetricCheck:
    task: str
    metric: str
    observed: Optional[float]
    min_threshold: Optional[float] = None
    reference: Optional[float] = None
    tolerance: Optional[float] = None
    error: Optional[str] = None  # set when the metric could not be read

    @property
    def ok(self) -> bool:
        if self.error is not None or self.observed is None:
            return False
        if self.min_threshold is not None and self.observed < self.min_threshold:
            return False
        if self.reference is not None and self.tolerance is not None:
            if abs(self.observed - self.reference) > self.tolerance:
                return False
        return True

    def describe(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        if self.error is not None:
            return f"[{status}] {self.task}/{self.metric}: {self.error}"
        parts = [f"observed={self.observed:.4f}"]
        if self.min_threshold is not None:
            parts.append(f"min={self.min_threshold:.4f}")
        if self.reference is not None and self.tolerance is not None:
            parts.append(f"ref={self.reference:.4f}±{self.tolerance:.4f}")
        return f"[{status}] {self.task}/{self.metric}: " + ", ".join(parts)


@dataclass
class SampleCheck:
    task: str
    expected: int
    observed: Optional[int]

    @property
    def ok(self) -> bool:
        return self.observed is not None and self.observed == self.expected

    def describe(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        obs = "unknown" if self.observed is None else str(self.observed)
        return f"[{status}] {self.task}: samples observed={obs} expected={self.expected}"


@dataclass
class GateReport:
    metric_checks: List[MetricCheck] = field(default_factory=list)
    sample_checks: List[SampleCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.metric_checks) and all(c.ok for c in self.sample_checks)

    def render(self) -> str:
        lines = [c.describe() for c in self.sample_checks]
        lines += [c.describe() for c in self.metric_checks]
        lines.append("")
        lines.append("GATE: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)

    def failures(self) -> List[str]:
        out = [c.describe() for c in self.sample_checks if not c.ok]
        out += [c.describe() for c in self.metric_checks if not c.ok]
        return out


def _metric_check(results: EvalResults, task: str, metric: str, thresholds: MetricThreshold) -> MetricCheck:
    check = MetricCheck(
        task=task,
        metric=metric,
        observed=None,
        min_threshold=thresholds.min,
        reference=thresholds.reference,
        tolerance=thresholds.tolerance,
    )
    observed = results.metric(task, metric)
    if observed is None:
        available = sorted(results.numeric_metrics(task))
        check.error = f"metric {metric!r} not in results[{task!r}]; available: {available}"
    else:
        check.observed = observed
    return check


def evaluate_gate(results: EvalResults, spec: GateSpec) -> GateReport:
    """Compare a run's results against a gate spec; return a :class:`GateReport`."""
    if not spec.tasks:
        raise ValueError("spec has no 'tasks' entries to check")

    report = GateReport()
    for task, task_spec in spec.tasks.items():
        if task_spec.expected_samples is not None:
            report.sample_checks.append(
                SampleCheck(task=task, expected=int(task_spec.expected_samples), observed=results.sample_count(task))
            )
        if not task_spec.metrics:
            raise ValueError(f"spec task {task!r} has no 'metrics' to check")
        for metric, thresholds in task_spec.metrics.items():
            report.metric_checks.append(_metric_check(results, task, metric, thresholds))
    return report


# --- recording a spec from a run ----------------------------------------------

# lm-eval's adapter name for chat-completions models; recorded in provenance and used
# to infer apply_chat_template when the recorder does not pass it explicitly.
_LOCAL_CHAT_COMPLETIONS = "local-chat-completions"

# Metric families we record floors for (exact_match / acc / pass@k); their stderr
# companions are skipped. pass@k is included so sampled tasks (AIME/MATH/HumanEval run
# with `-- --num_samples N`) can be recorded/gated too.
_GATED = ("exact_match", "acc", "pass@")


def _is_gated_metric(name: str) -> bool:
    return any(k in name for k in _GATED) and "stderr" not in name


def build_spec(
    results: EvalResults,
    tasks: List[str],
    margin: float,
    *,
    tolerance: Optional[float] = None,
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    tokenizer: Optional[str] = None,
    apply_chat_template: Optional[bool] = None,
    num_fewshot: Optional[int] = None,
    seed: Optional[int] = None,
) -> GateSpec:
    """Build a gate spec from a real run.

    Every gated metric gets a wide "model isn't broken/empty" floor,
    ``min = max(0.05, observed - margin)``. With ``tolerance`` set, it also gets a tight
    two-sided band ``reference = observed`` ± ``tolerance``, which is what actually
    catches a serving regression -- use it once a run is reproducible enough (greedy +
    ``num_concurrent=1``) that the band holds run-to-run. Provenance is read from the
    results file plus explicit overrides.
    """
    task_specs = {}
    for task in tasks:
        metrics = {}
        observed = {}
        for name, value in sorted(results.numeric_metrics(task).items()):
            if _is_gated_metric(name):
                rounded = round(value, 4)
                observed[name] = rounded
                floor = round(max(0.05, value - margin), 4)
                if tolerance is not None:
                    metrics[name] = MetricThreshold(min=floor, reference=rounded, tolerance=tolerance)
                else:
                    metrics[name] = MetricThreshold(min=floor)
        task_specs[task] = TaskSpec(
            metrics=metrics,
            observed=observed,
            expected_samples=results.sample_count(task),
        )
    resolved_model = model or results.model_name
    adapter = results.model_source
    if apply_chat_template is None and adapter is not None:
        apply_chat_template = adapter == _LOCAL_CHAT_COMPLETIONS
    recorded_seed = results.config.get("random_seed")
    provenance = SpecProvenance(
        model=resolved_model,
        model_revision=model_revision,
        tokenizer=tokenizer or resolved_model,
        lm_eval_version=results.lm_eval_version,
        adapter=adapter,
        apply_chat_template=apply_chat_template,
        limit=results.config.get("limit"),
        num_fewshot=num_fewshot,
        seed=recorded_seed if recorded_seed is not None else seed,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        note="Coarse smoke gate seeded from a real run.",
    )
    return GateSpec(provenance=provenance, tasks=task_specs)


# --- CLI ----------------------------------------------------------------------


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """Gate or record evalchemy regression specs."""


@cli.command()
@click.option("--results", "results_path", required=True, help="A results_*.json file OR a run dir to search.")
@click.option("--spec", "spec_path", default=DEFAULT_SPEC, help="Golden gate spec json.")
def check(results_path: str, spec_path: str) -> None:
    """Gate a run against a spec. Exit 0 = pass, 1 = fail."""
    if not spec_path or not os.path.exists(spec_path):
        raise click.UsageError(f"no spec to gate against ({spec_path!r}); pass --spec")
    report = evaluate_gate(EvalResults.load_path_or_dir(results_path), GateSpec.load(spec_path))
    click.echo(report.render())
    raise SystemExit(0 if report.ok else 1)


@cli.command()
@click.option("--results", "results_path", required=True, help="A results_*.json file OR a run dir to search.")
@click.option("--spec", "spec_path", default=DEFAULT_SPEC, help="Where to write the spec.")
@click.option("--model", default=None, help="Model id for provenance (default: read from results).")
@click.option("--model-revision", default=None, help="Model revision for provenance (pin it; the Hub tag is mutable).")
@click.option("--tokenizer", default=None, help="Tokenizer id for provenance (default: --model).")
@click.option("--num-fewshot", type=int, default=None, help="num_fewshot for provenance.")
@click.option("--seed", type=int, default=None, help="Seed for provenance (default: read from results).")
@click.option(
    "--apply-chat-template/--no-apply-chat-template",
    default=None,
    help="Record chat-template mode (default: infer from the results adapter).",
)
@click.option("--tasks", default=None, help="Comma-separated tasks to record (default: all tasks in the results).")
@click.option(
    "--margin",
    type=float,
    default=0.25,
    help="Floor headroom: min = max(0.05, observed - margin). Wide by default to absorb small-sample variance.",
)
@click.option(
    "--tolerance",
    type=float,
    default=None,
    help="Also emit a tight two-sided band reference=observed ± tolerance (the real regression gate). "
    "Set it to the run-to-run variance you measured; omit for a floor-only smoke spec.",
)
def record(
    results_path: str,
    spec_path: str,
    model: Optional[str],
    model_revision: Optional[str],
    tokenizer: Optional[str],
    num_fewshot: Optional[int],
    seed: Optional[int],
    apply_chat_template: Optional[bool],
    tasks: Optional[str],
    margin: float,
    tolerance: Optional[float],
) -> None:
    """Write a golden gate spec from a real run's results."""
    results = EvalResults.load_path_or_dir(results_path)
    task_list = tasks.split(",") if tasks else list(results.results.keys())
    spec = build_spec(
        results,
        task_list,
        margin,
        tolerance=tolerance,
        model=model,
        model_revision=model_revision,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        num_fewshot=num_fewshot,
        seed=seed,
    )
    spec.save(spec_path)
    click.echo(f"wrote spec -> {spec_path}")
    click.echo(spec.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    cli()
