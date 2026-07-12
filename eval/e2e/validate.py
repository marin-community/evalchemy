# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check an eval run against a golden baseline, or record a new baseline.

``check`` compares a run's ``results_*.json`` to a baseline and exits non-zero on
regression; ``record`` writes a baseline from a run. The gate is coarse (sample-count
+ a per-metric floor); see ``compare.py``.

    python -m eval.e2e.validate check  --results eval/e2e/runs/<ts> --baseline eval/e2e/baselines/qwen3-0.6b.json
    python -m eval.e2e.validate record --results eval/e2e/runs/<ts> --baseline eval/e2e/baselines/qwen3-0.6b.json
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import click

from eval.e2e.compare import evaluate_gate
from eval.e2e.models import (
    Baseline,
    BaselineProvenance,
    E2EConfig,
    EvalResults,
    MetricThreshold,
    TaskBaseline,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "e2e", "config.yaml")

# Metric families we record floors for (exact_match / acc / pass@k); never their
# stderr companions. pass@k is included so sampled tasks (AIME/MATH/HumanEval run
# with `-- --num_samples N`) can be recorded/gated too.
_GATED = ("exact_match", "acc", "pass@")


def _is_gated_metric(name: str) -> bool:
    return any(k in name for k in _GATED) and "stderr" not in name


def build_baseline(
    results: EvalResults,
    tasks: List[str],
    margin: float,
    cfg: E2EConfig,
    model: Optional[str],
) -> Baseline:
    """Build a floor-only smoke baseline from a real run.

    ``min = max(0.05, observed - margin)`` -- a wide "model isn't broken/empty"
    floor. A tight reference+tolerance band false-fails on small-sample variance
    (gsm8k strict-match swings ~3/20 at limit=20 even greedy), so it is not emitted;
    the observed values are kept for provenance. See README.
    """
    task_models = {}
    for task in tasks:
        metrics = {}
        observed = {}
        for name, value in sorted(results.numeric_metrics(task).items()):
            if _is_gated_metric(name):
                observed[name] = round(value, 4)
                metrics[name] = MetricThreshold(min=round(max(0.05, value - margin), 4))
        task_models[task] = TaskBaseline(
            metrics=metrics,
            observed=observed,
            expected_samples=results.sample_count(task),
        )
    provenance = BaselineProvenance(
        model=model or cfg.model or results.model_name,
        model_revision=cfg.model_revision,
        tokenizer=cfg.tokenizer or model or cfg.model,
        lm_eval_version=results.lm_eval_version,
        adapter=results.model_source,
        apply_chat_template=cfg.eval.apply_chat_template,
        limit=results.config.get("limit") if results.config.get("limit") is not None else cfg.eval.limit,
        num_fewshot=cfg.eval.num_fewshot,
        seed=results.config.get("random_seed") if results.config.get("random_seed") is not None else cfg.eval.seed,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        note="Smoke/coarse gate seeded from a real run -- not a statistical regression baseline.",
    )
    return Baseline(provenance=provenance, tasks=task_models)


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """Gate or record evalchemy e2e baselines."""


@cli.command()
@click.option("--results", "results_path", required=True, help="A results_*.json file OR a run dir to search.")
@click.option("--baseline", "baseline_path", default=None, help="Golden baseline json (default: config 'baseline').")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Config yaml (for the default baseline path).")
def check(results_path: str, baseline_path: Optional[str], config_path: str) -> None:
    """Gate a run against a baseline. Exit 0 = pass, 1 = fail."""
    cfg = E2EConfig.load_or_empty(config_path)
    baseline_path = baseline_path or cfg.baseline
    if not baseline_path or not os.path.exists(baseline_path):
        raise click.UsageError(f"no baseline to gate against ({baseline_path!r}); pass --baseline")
    report = evaluate_gate(EvalResults.load_path_or_dir(results_path), Baseline.load(baseline_path))
    click.echo(report.render())
    raise SystemExit(0 if report.ok else 1)


@cli.command()
@click.option("--results", "results_path", required=True, help="A results_*.json file OR a run dir to search.")
@click.option(
    "--baseline", "baseline_path", default=None, help="Where to write the baseline (default: config 'baseline')."
)
@click.option(
    "--config", "config_path", default=_DEFAULT_CONFIG, help="Config yaml (provenance + default baseline path)."
)
@click.option("--model", default=None, help="Model id for provenance (default: config 'model').")
@click.option("--tasks", default=None, help="Comma-separated tasks to record (default: all tasks in the results).")
@click.option(
    "--margin",
    type=float,
    default=0.25,
    help="Floor headroom: min = max(0.05, observed - margin). Wide by default to absorb small-sample variance.",
)
def record(
    results_path: str,
    baseline_path: Optional[str],
    config_path: str,
    model: Optional[str],
    tasks: Optional[str],
    margin: float,
) -> None:
    """Write a golden baseline from a real run's results."""
    cfg = E2EConfig.load_or_empty(config_path)
    baseline_path = baseline_path or cfg.baseline
    if not baseline_path:
        raise click.UsageError("no output path (--baseline or config 'baseline')")
    results = EvalResults.load_path_or_dir(results_path)
    task_list = tasks.split(",") if tasks else list(results.results.keys())
    baseline = build_baseline(results, task_list, margin, cfg, model)
    baseline.save(baseline_path)
    click.echo(f"wrote baseline -> {baseline_path}")
    click.echo(baseline.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    cli()
