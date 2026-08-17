# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run evalchemy against a served model and print the scores.

Brings up an inference server -- a TPU via ``marin-serve``, or an OpenAI-compatible
endpoint you already have -- runs ``python -m eval.eval`` against it, and prints each
task's metrics. Defaults live in ``eval/serve_eval/configs/qwen-tiny.yaml``; every field
is overridable.

    python -m eval.serve_eval.run --model Qwen/Qwen3-0.6B \
        --tpu v6e-4 --marin-workspace /path/to/marin

    python -m eval.serve_eval.run --provider endpoint --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Dict, List, Optional, Union

import click
from evalchemy_config import materialize_eval_args
from rigging import telemetry

from eval.serve_eval.config import RunConfig
from eval.serve_eval.providers import PHASE_DURATION, ServedModel, api_root, build_provider
from eval.serve_eval.results import EvalResults

logger = logging.getLogger("eval.serve_eval")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "serve_eval", "configs", "qwen-tiny.yaml")
_SHUTDOWN_TIMEOUT = 2.0
_DRAIN_POLL_INTERVAL = 0.01
_SNAPSHOT_GAUGE_ATTRIBUTES = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)
_WORK_COMPLETED = telemetry.gauge("work_completed", unit="{item}")
_SECONDS_PER_TRIAL = telemetry.gauge("seconds_per_trial", unit="s/trial")

# lm-eval OpenAI-compatible model backends (registry names it resolves via
# lm_eval.api.registry.get_model). Chat endpoint when a chat template is applied.
LOCAL_COMPLETIONS = "local-completions"
LOCAL_CHAT_COMPLETIONS = "local-chat-completions"
_ADAPTER_PATH = {LOCAL_COMPLETIONS: "completions", LOCAL_CHAT_COMPLETIONS: "chat/completions"}

ModelArgValue = Union[str, int, float, bool]


@dataclass
class _EvaluationPhase:
    duration_seconds: float = 0.0
    exit_code: int | None = None


class _EvaluationOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@contextmanager
def _measure_evaluation() -> Iterator[_EvaluationPhase]:
    phase = _EvaluationPhase()
    started = time.monotonic()
    outcome = _EvaluationOutcome.FAILED
    try:
        yield phase
    except KeyboardInterrupt:
        outcome = _EvaluationOutcome.INTERRUPTED
        raise
    else:
        outcome = _EvaluationOutcome.SUCCEEDED
    finally:
        phase.duration_seconds = time.monotonic() - started
        attributes = {"phase": "evaluation", "outcome": outcome}
        if phase.exit_code is not None:
            attributes["exit_code"] = str(phase.exit_code)
        PHASE_DURATION.record(phase.duration_seconds, attributes=attributes)


def configure_telemetry(
    endpoint: str | None,
    *,
    root_run_uid: str,
    execution_uid: str,
    model: str,
    provider: str,
    tasks: list[str],
    serving_job_id: str | None = None,
) -> None:
    """Configure telemetry once when a Finelog endpoint was provided."""
    if endpoint is None:
        return
    attributes = {
        "root_run_uid": root_run_uid,
        "execution_uid": execution_uid,
        "model": model,
        "provider": provider,
        "tasks": ",".join(tasks),
    }
    if serving_job_id is not None:
        attributes["serving_job_id"] = serving_job_id
    telemetry.configure(endpoint=endpoint, service="evalchemy", attributes=attributes)


def shutdown_telemetry() -> None:
    """Give telemetry one bounded opportunity to drain without changing the run."""
    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
    status = telemetry.runtime_status()
    while status.configured and status.queued_records:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_DRAIN_POLL_INTERVAL, remaining))
        status = telemetry.runtime_status()
    telemetry.shutdown(max(0.0, deadline - time.monotonic()))


def _record_completed_work(results: EvalResults, tasks: list[str], evaluation_duration: float) -> None:
    sample_counts = [results.sample_count(task) for task in tasks]
    for task, samples in zip(tasks, sample_counts, strict=True):
        if samples is not None:
            _WORK_COMPLETED.set(
                samples,
                attributes={"work_kind": "trial", "scope": "task", "task": task, **_SNAPSHOT_GAUGE_ATTRIBUTES},
            )
    if all(samples is not None for samples in sample_counts):
        total_trials = sum(samples for samples in sample_counts if samples is not None)
        _WORK_COMPLETED.set(
            total_trials,
            attributes={"work_kind": "trial", "scope": "run", **_SNAPSHOT_GAUGE_ATTRIBUTES},
        )
        if total_trials > 0:
            _SECONDS_PER_TRIAL.set(
                evaluation_duration / total_trials,
                attributes={"source_phase": "evaluation", **_SNAPSHOT_GAUGE_ATTRIBUTES},
            )


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
        # Loglikelihood scoring slices echoed logprobs at lm-eval's local token
        # boundary. The completions endpoint must therefore receive those exact
        # token IDs; a text round trip can retokenize a leading-space target.
        # Chat requests are message objects and stay text-based.
        "tokenizer_backend": "huggingface",
        "tokenized_requests": adapter == LOCAL_COMPLETIONS,
    }
    if served.api_key is not None:
        args["api_key"] = served.api_key
    if served.tokenizer is not None:
        args["tokenizer"] = served.tokenizer
    if extra:
        args.update(extra)
    return ",".join(f"{k}={_model_arg(v)}" for k, v in args.items())


def build_eval_argv(
    served: ServedModel,
    cfg: RunConfig,
    output_dir: str,
    limit: Optional[int],
    extra_args: Sequence[str],
    python: str,
) -> List[str]:
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
    ]
    # ``limit`` is post-processed because non-positive runner values mean the full
    # task, whereas the portable config preserves the original requested value.
    argv += materialize_eval_args(cfg.evaluation.model_copy(update={"limit": limit}))
    argv += [
        "--output_path",
        output_dir,
        "--log_samples",
    ]
    argv += list(extra_args)  # verbatim eval.eval passthrough (last => can override)
    return argv


def run_eval(argv: List[str]) -> float:
    """Run the evaluation child and return its critical-path wall time in seconds."""
    logger.info("running eval.eval:\n  %s", " ".join(argv))
    with _measure_evaluation() as phase:
        result = subprocess.run(argv, cwd=_REPO_ROOT)  # noqa: S603 - operator-supplied args
        phase.exit_code = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f"eval.eval exited with code {result.returncode}")
    return phase.duration_seconds


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
@click.option(
    "--telemetry-endpoint",
    default=None,
    envvar="FINELOG_TELEMETRY_ENDPOINT",
    help="Finelog /v1/telemetry endpoint. Unset disables telemetry.",
)
@click.option(
    "--root-run-uid",
    default=None,
    envvar="EVAL_ROOT_RUN_UID",
    help="Stable identity for the logical eval effort (default: UUID).",
)
@click.option(
    "--execution-uid",
    default=None,
    envvar="EVAL_EXECUTION_UID",
    help="Identity for this invocation or retry (default: UUID).",
)
@click.option(
    "--serving-job-id",
    default=None,
    envvar="EVAL_SERVING_JOB_ID",
    help="Optional canonical Iris serving job ID for joining service=vllm telemetry.",
)
# endpoint provider
@click.option("--base-url", default=None, envvar="E2E_BASE_URL", help="OpenAI /v1 root (endpoint provider).")
@click.option("--api-key", default=None, envvar="E2E_API_KEY", help="Bearer token for the endpoint.")
@click.option("--no-wait-ready", is_flag=True, help="Skip the /v1/models readiness poll.")
# marin-serve provider
@click.option("--cluster", default=None, help="Iris cluster (marin-serve provider).")
@click.option(
    "--tpu",
    default=None,
    help="TPU slice type (marin-serve provider).",
)
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
    telemetry_endpoint: Optional[str],
    root_run_uid: Optional[str],
    execution_uid: Optional[str],
    serving_job_id: Optional[str],
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
    extra_eval_args: tuple[str, ...],
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
    root_run_uid = root_run_uid or str(uuid.uuid4())
    execution_uid = execution_uid or str(uuid.uuid4())
    configure_telemetry(
        telemetry_endpoint,
        root_run_uid=root_run_uid,
        execution_uid=execution_uid,
        serving_job_id=serving_job_id,
        model=cfg.model,
        provider=provider,
        tasks=cfg.tasks,
    )
    try:
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
            logger.info(
                "served model: base_url=%s model=%s (auth=%s)",
                served.base_url,
                served.model,
                bool(served.api_key),
            )
            evaluation_duration = run_eval(
                build_eval_argv(served, cfg, output_dir, limit, extra_eval_args, python_bin)
            )

        results_path = EvalResults.find_latest_path(output_dir)
        results = EvalResults.load(results_path)
        _record_completed_work(results, cfg.tasks, evaluation_duration)
        click.echo("\n" + summarize(results, cfg.tasks))
        click.echo(f"\nresults: {results_path}")
        click.echo(f"to gate: python -m eval.regression.validate check --results {output_dir} --spec <spec.json>")
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
