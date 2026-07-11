# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""One-shot e2e: serve a model (Marin) -> evaluate with evalchemy -> gate vs baseline.

    python -m eval.e2e.run_e2e --provider endpoint \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-0.6B

    python -m eval.e2e.run_e2e --provider marin-serve \
        --model Qwen/Qwen3-0.6B --cluster marin --tpu v6e-8

Config defaults live in ``eval/e2e/config.yaml``; every field is overridable on
the CLI. See ``eval/e2e/README.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eval.e2e.compare import (
    effective_sample_count,
    evaluate_gate,
    find_latest_results,
    load_results,
)
from eval.e2e.eval_args import EvalInvocation, ServedModel, build_eval_argv
from eval.e2e.providers import build_provider

logger = logging.getLogger("eval.e2e")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "eval", "e2e", "config.yaml")


def _load_config(path: str) -> Dict[str, Any]:
    import yaml  # local import: not needed by the pure eval_args/compare core

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalchemy end-to-end test against a served model.")
    # Default provisions the accelerator itself via marin-serve; `endpoint`
    # attaches to a server you already have.
    p.add_argument("--provider", choices=["marin-serve", "endpoint"], default="marin-serve")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to e2e config yaml.")
    p.add_argument("--model", default=None, help="HF model id (or gs:// path) to serve/evaluate.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer id for lm-eval (defaults to --model).")
    p.add_argument("--tasks", default=None, help="Comma-separated task list (overrides config).")
    p.add_argument("--limit", type=int, default=None, help="Per-task sample cap (the 'sample query set').")
    p.add_argument("--baseline", default=None, help="Baseline json to gate against (overrides config).")
    p.add_argument("--output-dir", default=None, help="Where eval.eval writes results (default: temp under runs/).")
    p.add_argument("--python", default=sys.executable, help="Python used to run eval.eval.")
    # endpoint provider
    p.add_argument("--base-url", default=os.environ.get("E2E_BASE_URL"), help="OpenAI /v1 root (endpoint provider).")
    p.add_argument("--api-key", default=os.environ.get("E2E_API_KEY"), help="Bearer token for the endpoint.")
    p.add_argument("--no-wait-ready", action="store_true", help="Skip the /v1/models readiness poll.")
    # marin-serve provider
    p.add_argument("--cluster", default=None, help="Iris cluster (marin-serve provider).")
    p.add_argument("--tpu", default=None, help="TPU slice type (marin-serve provider).")
    p.add_argument("--name", default=None, help="Iris job name (marin-serve provider).")
    p.add_argument(
        "--access",
        choices=["link", "private"],
        default=None,
        help="marin-serve proxy access. 'link' (default) mints a PUBLIC capability URL; "
        "'private' restricts to cluster identity.",
    )
    p.add_argument("--region", default=None, help="Region(s) to pin the TPU slice to, e.g. europe-west4.")
    p.add_argument(
        "--marin-workspace",
        default=os.environ.get("MARIN_WORKSPACE"),
        help="marin checkout to run marin-serve from (it bundles cwd as the Iris job workspace).",
    )
    p.add_argument("--wait-timeout", type=float, default=None, help="Seconds for vLLM to boot (marin-serve).")
    p.add_argument("--timeout-hours", type=float, default=None, help="Server self-stop backstop (marin-serve).")
    # modes
    p.add_argument(
        "--record-baseline", action="store_true", help="Write observed results as a baseline instead of gating."
    )
    p.add_argument(
        "--record-margin",
        type=float,
        default=0.25,
        help="Floor headroom when recording: min = max(0.05, observed - margin). Wide by "
        "default to absorb small-sample variance (the gate is a smoke check).",
    )
    p.add_argument("--skip-eval", action="store_true", help="Only gate an existing --output-dir (no serve/eval).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _build_invocation(
    cfg: Dict[str, Any], args: argparse.Namespace, served: ServedModel, output_dir: str
) -> EvalInvocation:
    tasks = args.tasks.split(",") if args.tasks else _deep_get(cfg, "eval", "tasks", default=["gsm8k"])
    limit = args.limit if args.limit is not None else _deep_get(cfg, "eval", "limit")
    extra = dict(_deep_get(cfg, "eval", "extra_model_args", default={}) or {})
    return EvalInvocation(
        served=served,
        tasks=list(tasks),
        output_path=output_dir,
        apply_chat_template=bool(_deep_get(cfg, "eval", "apply_chat_template", default=False)),
        limit=limit,
        num_fewshot=_deep_get(cfg, "eval", "num_fewshot"),
        batch_size=_deep_get(cfg, "eval", "batch_size", default=1),
        seed=_deep_get(cfg, "eval", "seed", default=1234),
        gen_kwargs=_deep_get(cfg, "eval", "gen_kwargs"),
        extra_model_args=extra,
    )


def _run_eval(inv: EvalInvocation, python: str) -> None:
    argv = build_eval_argv(inv, python=python)
    logger.info("running eval.eval:\n  %s", " ".join(argv))
    result = subprocess.run(argv, cwd=_REPO_ROOT)  # noqa: S603 - operator-supplied args
    if result.returncode != 0:
        raise RuntimeError(f"eval.eval exited with code {result.returncode}")


def _record_baseline(results: dict, cfg: Dict[str, Any], args: argparse.Namespace, inv: EvalInvocation) -> dict:
    """Build a baseline dict from observed results (provenance + floors)."""
    tasks_out: Dict[str, Any] = {}
    for task in inv.tasks:
        task_results = (results.get("results") or {}).get(task, {})
        metrics: Dict[str, Any] = {}
        observed: Dict[str, float] = {}
        for metric, value in sorted(task_results.items()):
            # Gate on quality metrics only -- never on their stderr companions
            # (``exact_match_stderr,*`` contains "exact_match" as a substring).
            if (
                isinstance(value, (int, float))
                and ("exact_match" in metric or "acc" in metric)
                and "stderr" not in metric
            ):
                obs = float(value)
                observed[metric] = round(obs, 4)
                # Floor-only smoke gate. A tight reference+/-tolerance band
                # false-fails on small-sample (limit=20) variance -- gsm8k
                # strict-match alone swings ~3/20 run-to-run even with greedy
                # decoding. The sample-count check plus this conservative "model
                # isn't broken/empty" floor are the robust smoke signals (see
                # README). compare.py still supports an optional reference+tolerance
                # band for anyone wanting a tighter regression gate at higher limit.
                metrics[metric] = {"min": round(max(0.05, obs - args.record_margin), 4)}
        entry: Dict[str, Any] = {"metrics": metrics, "observed": observed}
        n = effective_sample_count(results, task)
        if n is not None:
            entry["expected_samples"] = n
        tasks_out[task] = entry
    return {
        "provenance": {
            "model": args.model or _deep_get(cfg, "model"),
            "model_revision": _deep_get(cfg, "model_revision"),
            "tokenizer": inv.served.tokenizer,
            "lm_eval_version": results.get("lm_eval_version") or (results.get("config") or {}).get("lm_eval_version"),
            "serving_backend": args.provider,
            "adapter": inv.adapter,
            "apply_chat_template": inv.apply_chat_template,
            "limit": inv.limit,
            "num_fewshot": inv.num_fewshot,
            "seed": inv.seed,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "note": "Smoke/coarse gate seeded from a real run -- not a statistical regression baseline.",
        },
        "tasks": tasks_out,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[e2e] %(message)s",
    )
    cfg = _load_config(args.config) if os.path.exists(args.config) else {}

    model = args.model or _deep_get(cfg, "model")
    if not model:
        logger.error("no model given (--model or config 'model')")
        return 2
    args.model = model
    baseline_path = args.baseline or _deep_get(cfg, "baseline")

    output_dir = args.output_dir or os.path.join(
        _REPO_ROOT, "eval", "e2e", "runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    os.makedirs(output_dir, exist_ok=True)

    # --skip-eval only re-gates an existing output dir (no serving).
    if args.skip_eval:
        served = ServedModel(
            base_url=args.base_url or "http://unused/v1", model=model, tokenizer=args.tokenizer or model
        )
        inv = _build_invocation(cfg, args, served, output_dir)
        return _finish(cfg, args, inv, baseline_path)

    prov_cfg_key = args.provider
    provider = build_provider(
        args.provider,
        model,
        base_url=args.base_url,
        api_key=args.api_key,
        tokenizer=args.tokenizer or _deep_get(cfg, "tokenizer"),
        cluster=args.cluster or _deep_get(cfg, "provider", prov_cfg_key, "cluster", default="marin"),
        tpu=args.tpu or _deep_get(cfg, "provider", prov_cfg_key, "tpu", default="v6e-8"),
        name=args.name,
        access=args.access or _deep_get(cfg, "provider", prov_cfg_key, "access", default="link"),
        region=args.region or _deep_get(cfg, "provider", prov_cfg_key, "region"),
        marin_workspace=args.marin_workspace or _deep_get(cfg, "provider", prov_cfg_key, "marin_workspace"),
        wait_timeout_s=args.wait_timeout or _deep_get(cfg, "provider", prov_cfg_key, "wait_timeout_s", default=1800.0),
        timeout_hours=args.timeout_hours or _deep_get(cfg, "provider", prov_cfg_key, "timeout_hours", default=2.0),
        wait_ready=not args.no_wait_ready,
    )

    with provider as served:
        logger.info("served model: base_url=%s model=%s (auth=%s)", served.base_url, served.model, bool(served.api_key))
        inv = _build_invocation(cfg, args, served, output_dir)
        _run_eval(inv, args.python)
        return _finish(cfg, args, inv, baseline_path)


def _finish(cfg: Dict[str, Any], args: argparse.Namespace, inv: EvalInvocation, baseline_path: Optional[str]) -> int:
    results_path = find_latest_results(inv.output_path)
    logger.info("results: %s", results_path)
    results = load_results(results_path)

    if args.record_baseline:
        if not baseline_path:
            logger.error("--record-baseline needs a --baseline path (or config 'baseline')")
            return 2
        baseline = _record_baseline(results, cfg, args, inv)
        os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, sort_keys=False)
            f.write("\n")
        logger.info("wrote baseline -> %s\n%s", baseline_path, json.dumps(baseline, indent=2))
        return 0

    if not baseline_path or not os.path.exists(baseline_path):
        logger.error("no baseline to gate against (%r); pass --baseline or --record-baseline first", baseline_path)
        return 2

    report = evaluate_gate(results, json.load(open(baseline_path, encoding="utf-8")))
    print("\n" + report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
