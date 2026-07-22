# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run evalchemy against a served model and print the scores.

Brings up an inference server -- a TPU via ``marin-serve``, or an OpenAI-compatible
endpoint you already have -- runs ``python -m eval.eval`` against it, and prints each
task's metrics. Defaults live in ``eval/serve_eval/configs/qwen-tiny.yaml``; every field
is overridable.

    python -m eval.serve_eval.run --model Qwen/Qwen3-0.6B \
        --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

    python -m eval.serve_eval.run --provider endpoint --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import click

from eval.serve_eval.config import RunConfig
from eval.serve_eval.providers import ServedModel, api_root, build_provider
from eval.serve_eval.results import EvalResults

logger = logging.getLogger("eval.serve_eval")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "serve_eval", "configs", "qwen-tiny.yaml")

# lm-eval OpenAI-compatible model backends (registry names it resolves via
# lm_eval.api.registry.get_model). Chat endpoint when a chat template is applied.
LOCAL_COMPLETIONS = "local-completions"
LOCAL_CHAT_COMPLETIONS = "local-chat-completions"
_ADAPTER_PATH = {LOCAL_COMPLETIONS: "completions", LOCAL_CHAT_COMPLETIONS: "chat/completions"}

ModelArgValue = Union[str, int, float, bool]


def adapter_for(apply_chat_template: bool) -> str:
    return LOCAL_CHAT_COMPLETIONS if apply_chat_template else LOCAL_COMPLETIONS


def endpoint_url(base_url: str, adapter: str) -> str:
    """The concrete OpenAI endpoint URL for ``adapter`` under the ``/v1`` root."""
    return api_root(base_url) + "/" + _ADAPTER_PATH[adapter]


def _model_arg(value: ModelArgValue) -> str:
    # lm-eval parses --model_args as comma-separated key=value, so a value cannot
    # contain a comma or the pair boundary is lost.
    text = "True" if value is True else "False" if value is False else str(value)
    if "," in text:
        raise ValueError(f"model_args value cannot contain ',': {text!r}")
    return text


def build_model_args(served: ServedModel, adapter: str, extra: Optional[Dict[str, ModelArgValue]] = None) -> str:
    """Build the comma-delimited ``--model_args`` string for a served endpoint."""
    args: Dict[str, ModelArgValue] = {
        "model": served.model,
        "base_url": endpoint_url(served.base_url, adapter),
        # The served model already tokenizes; keep requests as text and let lm-eval
        # use a HF tokenizer only for length bookkeeping.
        "tokenizer_backend": "huggingface",
        "tokenized_requests": False,
    }
    if served.api_key is not None:
        args["api_key"] = served.api_key
    if served.tokenizer is not None:
        args["tokenizer"] = served.tokenizer
    if extra:
        args.update(extra)
    return ",".join(f"{k}={_model_arg(v)}" for k, v in args.items())


def build_eval_argv(served: ServedModel, cfg: RunConfig, output_dir: str, limit, extra_args, python: str) -> List[str]:
    """Build the ``python -m eval.eval`` argv for this run.

    Uses the bare ``--apply_chat_template`` flag: that parser option is
    ``nargs="?", const=True`` (eval/lm_eval_compat.py), so a following value would be
    read as a chat-template name, not the boolean.
    """
    if not cfg.tasks:
        raise ValueError("no tasks to run (--tasks or config 'tasks')")
    adapter = adapter_for(cfg.apply_chat_template)
    argv = [
        python,
        "-m",
        "eval.eval",
        "--model",
        adapter,
        "--model_args",
        build_model_args(served, adapter, dict(cfg.extra_model_args)),
        "--tasks",
        ",".join(cfg.tasks),
        "--output_path",
        output_dir,
        "--log_samples",
    ]
    if cfg.apply_chat_template:
        argv.append("--apply_chat_template")  # bare flag => const=True
    if limit is not None:
        argv += ["--limit", str(limit)]
    if cfg.num_fewshot is not None:
        argv += ["--num_fewshot", str(cfg.num_fewshot)]
    if cfg.batch_size is not None:
        argv += ["--batch_size", str(cfg.batch_size)]
    if cfg.seed is not None:
        # evalchemy/lm-eval accept a 4-tuple "python,numpy,torch,fewshot"; a single
        # value sets all four (see eval/eval.py seed handling).
        argv += ["--seed", str(cfg.seed)]
    if cfg.gen_kwargs:
        argv += ["--gen_kwargs", cfg.gen_kwargs]
    argv += list(extra_args)  # verbatim eval.eval passthrough (last => can override)
    return argv


def run_eval(argv: List[str]) -> None:
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
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Path to the run config yaml.")
@click.option("--model", default=None, help="HF model id (or gs:// path) to serve/evaluate.")
@click.option("--tokenizer", default=None, help="Tokenizer id for lm-eval (defaults to --model).")
@click.option("--tasks", default=None, help="Comma-separated task list (overrides config).")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Per-task sample cap. Default: config 'limit'. Pass 0 (or negative) to run the FULL task.",
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
    region: Optional[str],
    marin_workspace: Optional[str],
    wait_timeout: Optional[float],
    timeout_hours: Optional[float],
    verbose: bool,
    extra_eval_args: tuple,
) -> None:
    """Serve a model, run the eval, and print the results.

    Anything after ``--`` is forwarded verbatim to ``python -m eval.eval`` -- e.g.
    ``run --tasks MATH500 -- --num_samples 8 --pass_at_k 1,8`` for pass@k.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[e2e] %(message)s")
    cfg = RunConfig.load(
        config_path,
        model=model,
        tokenizer=tokenizer,
        tasks=tasks.split(",") if tasks else None,
        limit=limit,
        cluster=cluster,
        tpu=tpu,
        region=region,
        marin_workspace=marin_workspace,
        wait_timeout_s=wait_timeout,
        timeout_hours=timeout_hours,
    )
    if not cfg.model:
        raise click.UsageError("no model given (--model or config 'model')")
    # --limit 0 (or negative) means the FULL task: lm-eval reads no --limit as "all samples".
    limit = None if (cfg.limit is not None and cfg.limit <= 0) else cfg.limit

    output_dir = output_dir or os.path.join(
        _REPO_ROOT, "eval", "serve_eval", "runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        region=cfg.region,
        marin_workspace=cfg.marin_workspace,
        wait_timeout_s=cfg.wait_timeout_s,
        timeout_hours=cfg.timeout_hours,
        wait_ready=not no_wait_ready,
    )

    with prov as served:
        logger.info("served model: base_url=%s model=%s (auth=%s)", served.base_url, served.model, bool(served.api_key))
        run_eval(build_eval_argv(served, cfg, output_dir, limit, extra_eval_args, python_bin))

    results_path = EvalResults.find_latest_path(output_dir)
    results = EvalResults.load(results_path)
    click.echo("\n" + summarize(results, cfg.tasks))
    click.echo(f"\nresults: {results_path}")
    click.echo(f"to gate: python -m eval.regression.validate check --results {output_dir} --spec <spec.json>")


if __name__ == "__main__":
    main()
