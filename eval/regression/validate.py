# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check an eval run against a regression gate spec, or record a new spec.

``check`` compares a run's ``results_*.json`` to a spec and exits non-zero on
regression; ``record`` writes a spec from a run. The gate is coarse (sample-count +
a per-metric floor); see ``gate.py``. Its inputs are a results file and a spec file.

    python -m eval.regression.validate check  --results <run-dir> --spec eval/regression/specs/qwen3-0.6b.json
    python -m eval.regression.validate record --results <run-dir> --spec eval/regression/specs/qwen3-0.6b.json
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import click

from eval.regression.gate import GateSpec, MetricThreshold, SpecProvenance, TaskSpec, evaluate_gate
from eval.serve_eval.results import EvalResults

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SPEC = os.path.join(_REPO_ROOT, "eval", "regression", "specs", "qwen3-0.6b.json")

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
    ``min = max(0.05, observed - margin)``. With ``tolerance`` set, it ALSO gets a tight
    two-sided band ``reference = observed`` ± ``tolerance``, which is what actually
    catches a serving regression -- use it once a run is reproducible enough (greedy +
    ``num_concurrent=1``) that the band holds run-to-run. Provenance is read from the
    results file plus explicit overrides, as context for whoever reads the spec.
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
