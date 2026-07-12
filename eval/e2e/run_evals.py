# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run evalchemy against a served model and print the scores.

Brings up an inference server -- a TPU via ``marin-serve``, or an OpenAI-compatible
endpoint you already have -- runs ``python -m eval.eval`` against it, and prints each
task's metrics. Defaults live in ``eval/e2e/qwen-tiny.yaml``; every field is overridable.

    python -m eval.e2e.run_evals --model Qwen/Qwen3-0.6B \
        --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

    python -m eval.e2e.run_evals --provider endpoint --base-url http://localhost:8000/v1
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
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "e2e", "qwen-tiny.yaml")


def build_invocation(cfg: E2EConfig, served: ServedModel, output_dir: str, limit, extra_args) -> EvalInvocation:
    return EvalInvocation(
        served=served,
        tasks=list(cfg.tasks),
        output_path=output_dir,
        apply_chat_template=cfg.apply_chat_template,
        limit=limit,
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        gen_kwargs=cfg.gen_kwargs,
        extra_model_args=dict(cfg.extra_model_args),
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
@click.option("--base-url", default=None, envvar="E2E_BASE_URL", help="OpenAI /v1 root (endpoint provider).")
@click.option("--api-key", default=None, envvar="E2E_API_KEY", help="Bearer token for the endpoint.")
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
    default=None,
    envvar="MARIN_WORKSPACE",
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
    cfg = E2EConfig.load(
        config_path,
        model=model,
        tokenizer=tokenizer,
        tasks=tasks.split(",") if tasks else None,
        limit=limit,
        cluster=cluster,
        tpu=tpu,
        region=region,
        access=access,
        marin_workspace=marin_workspace,
        wait_timeout_s=wait_timeout,
        timeout_hours=timeout_hours,
    )
    if not cfg.model:
        raise click.UsageError("no model given (--model or config 'model')")
    # --limit 0 (or negative) means the FULL task: lm-eval reads no --limit as "all samples".
    limit = None if (cfg.limit is not None and cfg.limit <= 0) else cfg.limit

    output_dir = output_dir or os.path.join(
        _REPO_ROOT, "eval", "e2e", "runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    os.makedirs(output_dir, exist_ok=True)

    prov = build_provider(
        provider,
        cfg.model,
        base_url=base_url,
        api_key=api_key,
        tokenizer=cfg.tokenizer,
        cluster=cfg.cluster,
        tpu=cfg.tpu,
        name=name,
        access=cfg.access,
        region=cfg.region,
        marin_workspace=cfg.marin_workspace,
        wait_timeout_s=cfg.wait_timeout_s,
        timeout_hours=cfg.timeout_hours,
        wait_ready=not no_wait_ready,
    )

    with prov as served:
        logger.info("served model: base_url=%s model=%s (auth=%s)", served.base_url, served.model, bool(served.api_key))
        inv = build_invocation(cfg, served, output_dir, limit, extra_eval_args)
        run_eval(inv, python_bin)

    results_path = EvalResults.find_latest_path(output_dir)
    results = EvalResults.load(results_path)
    click.echo("\n" + summarize(results, cfg.tasks))
    click.echo(f"\nresults: {results_path}")
    click.echo(f"to gate: python -m eval.e2e.validate check --results {output_dir} --baseline <baseline.json>")


if __name__ == "__main__":
    main()
