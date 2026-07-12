# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run an evalchemy eval against a served model, and print the results.

The **human-facing** half of the e2e harness: bring up an inference server (or
attach to one you already have), run the eval, and show what the model scored.
It does NOT pass/fail against a baseline -- that is ``eval.e2e.validate`` (the CI
half), so this stays a "just tell me the numbers" tool.

    # Provision a TPU via marin-serve, eval, print results:
    python -m eval.e2e.run_evals --model Qwen/Qwen3-0.6B \
        --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

    # Attach to a running OpenAI-compatible endpoint (no Marin needed):
    python -m eval.e2e.run_evals --provider endpoint --base-url http://localhost:8000/v1

Config defaults live in ``eval/e2e/config.yaml``; every field is overridable here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional

import click

from eval.e2e.eval_args import EvalInvocation, ServedModel, build_eval_argv
from eval.e2e.models import E2EConfig, EvalResults
from eval.e2e.providers import build_provider

logger = logging.getLogger("eval.e2e")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "e2e", "config.yaml")


def _pick(cli_value, config_value, default=None):
    """CLI override wins, then config, then default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def resolve_limit(cli_limit: Optional[int], config_limit: Optional[int], default: int = 200) -> Optional[int]:
    """Resolve the per-task sample cap into what lm-eval should receive.

    CLI value wins, else the config value, else ``default`` (a real-ish eval). The
    non-obvious part: ``<= 0`` is the escape hatch for the FULL task -- lm-eval
    reads a *missing* ``--limit`` as "all samples", so a non-positive request maps
    to ``None`` (omit the flag) rather than a literal cap of 0.
    """
    limit = _pick(cli_limit, config_limit, default)
    return None if limit <= 0 else limit


def build_invocation(cfg: E2EConfig, served: ServedModel, output_dir: str, tasks, limit, extra_args) -> EvalInvocation:
    ev = cfg.eval
    return EvalInvocation(
        served=served,
        tasks=list(tasks),
        output_path=output_dir,
        apply_chat_template=ev.apply_chat_template,
        limit=limit,
        num_fewshot=ev.num_fewshot,
        batch_size=ev.batch_size,
        seed=ev.seed,
        gen_kwargs=ev.gen_kwargs,
        extra_model_args=dict(ev.extra_model_args),
        extra_args=list(extra_args),
    )


def run_eval(inv: EvalInvocation, python: str) -> None:
    argv = build_eval_argv(inv, python=python)
    logger.info("running eval.eval:\n  %s", " ".join(argv))
    result = subprocess.run(argv, cwd=_REPO_ROOT)  # noqa: S603 - operator-supplied args
    if result.returncode != 0:
        raise RuntimeError(f"eval.eval exited with code {result.returncode}")


def summarize(results: EvalResults, tasks: List[str]) -> str:
    """A compact human table of each task's numeric metrics + sample count."""
    lines = []
    for task in tasks:
        n = results.sample_count(task)
        lines.append(f"{task}  (samples: {n if n is not None else '?'})")
        metrics = results.numeric_metrics(task)
        for name in sorted(metrics):
            if "stderr" in name:
                continue
            lines.append(f"    {name:<32} {metrics[name]:.4f}")
    return "\n".join(lines)


@click.command(context_settings={"show_default": True})
@click.option("--provider", type=click.Choice(["marin-serve", "endpoint"]), default="marin-serve")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Path to e2e config yaml.")
@click.option("--model", default=None, help="HF model id (or gs:// path) to serve/evaluate.")
@click.option("--tokenizer", default=None, help="Tokenizer id for lm-eval (defaults to --model).")
@click.option("--tasks", default=None, help="Comma-separated task list (overrides config).")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Per-task sample cap. Default: config 'limit' (200) -- a real-ish eval; CI passes 20 for a fast smoke. "
    "Pass 0 (or negative) to run the FULL task.",
)
@click.option("--output-dir", default=None, help="Where eval.eval writes results (default: a stamped dir under runs/).")
@click.option("--python", "python_bin", default=sys.executable, help="Python used to run eval.eval.")
# endpoint provider
@click.option("--base-url", default=lambda: os.environ.get("E2E_BASE_URL"), help="OpenAI /v1 root (endpoint provider).")
@click.option("--api-key", default=lambda: os.environ.get("E2E_API_KEY"), help="Bearer token for the endpoint.")
@click.option("--no-wait-ready", is_flag=True, help="Skip the /v1/models readiness poll.")
# marin-serve provider
@click.option("--cluster", default=None, help="Iris cluster (marin-serve provider).")
@click.option("--tpu", default=None, help="TPU slice type (marin-serve provider).")
@click.option("--name", default=None, help="Iris job name (marin-serve provider).")
@click.option(
    "--access",
    type=click.Choice(["link", "private"]),
    default=None,
    help="marin-serve proxy access. 'link' (default) mints a PUBLIC capability URL; 'private' is cluster-only.",
)
@click.option("--region", default=None, help="Region(s) to pin the TPU slice to, e.g. europe-west4.")
@click.option(
    "--marin-workspace",
    default=lambda: os.environ.get("MARIN_WORKSPACE"),
    help="marin checkout to run marin-serve from (it bundles cwd as the Iris job workspace).",
)
@click.option("--wait-timeout", type=float, default=None, help="Seconds for vLLM to boot (marin-serve).")
@click.option("--timeout-hours", type=float, default=None, help="Server self-stop backstop (marin-serve).")
@click.option("-v", "--verbose", is_flag=True)
@click.argument("extra_eval_args", nargs=-1, type=click.UNPROCESSED)
def main(
    provider: str,
    config_path: str,
    model: Optional[str],
    tokenizer: Optional[str],
    tasks: Optional[str],
    limit: Optional[int],
    output_dir: Optional[str],
    python_bin: str,
    base_url: Optional[str],
    api_key: Optional[str],
    no_wait_ready: bool,
    cluster: Optional[str],
    tpu: Optional[str],
    name: Optional[str],
    access: Optional[str],
    region: Optional[str],
    marin_workspace: Optional[str],
    wait_timeout: Optional[float],
    timeout_hours: Optional[float],
    verbose: bool,
    extra_eval_args: tuple,
) -> None:
    """Serve a model, run the eval, and print the results (no pass/fail gate).

    Anything after ``--`` is forwarded verbatim to ``python -m eval.eval`` -- e.g.
    ``run_evals --tasks MATH500 -- --num_samples 8 --pass_at_k 1,8`` for pass@k.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[e2e] %(message)s")
    cfg = E2EConfig.load_or_empty(config_path)

    model = model or cfg.model
    if not model:
        raise click.UsageError("no model given (--model or config 'model')")
    task_list = tasks.split(",") if tasks else list(cfg.eval.tasks)
    limit = resolve_limit(limit, cfg.eval.limit)

    output_dir = output_dir or os.path.join(
        _REPO_ROOT, "eval", "e2e", "runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    os.makedirs(output_dir, exist_ok=True)

    ms = cfg.provider.marin_serve
    prov = build_provider(
        provider,
        model,
        base_url=base_url,
        api_key=api_key,
        tokenizer=_pick(tokenizer, cfg.tokenizer),
        cluster=_pick(cluster, ms.cluster, "marin"),
        tpu=_pick(tpu, ms.tpu, "v5litepod-8"),
        name=name,
        access=_pick(access, ms.access, "link"),
        region=_pick(region, ms.region),
        marin_workspace=_pick(marin_workspace, ms.marin_workspace),
        wait_timeout_s=_pick(wait_timeout, ms.wait_timeout_s, 1800.0),
        timeout_hours=_pick(timeout_hours, ms.timeout_hours, 2.0),
        wait_ready=not no_wait_ready,
    )

    with prov as served:
        logger.info("served model: base_url=%s model=%s (auth=%s)", served.base_url, served.model, bool(served.api_key))
        inv = build_invocation(cfg, served, output_dir, task_list, limit, extra_eval_args)
        run_eval(inv, python_bin)

    results_path = EvalResults.find_latest_path(output_dir)
    results = EvalResults.load(results_path)
    click.echo("\n" + summarize(results, task_list))
    click.echo(f"\nresults: {results_path}")
    click.echo(f"to gate: python -m eval.e2e.validate check --results {output_dir} --baseline <baseline.json>")


if __name__ == "__main__":
    main()
