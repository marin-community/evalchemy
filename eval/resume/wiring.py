"""Auto-detect wiring — construct the ResumeManager from CLI run inputs (Stage 4).

This is the ONE construction site that feeds all three resume paths (global
invariant #5 — one interface). ``cli_evaluate`` (``eval/eval.py``) calls
:func:`build_resume_wiring` once after model init; the result both

  * sets ``args.resume_manager_factory`` — the seam the lm-eval-native (3b) and
    native pass@k (3c) call sites read (``resume_manager_factory(task_name) ->
    ResumeManager``), and
  * is used to ``attach_resume_manager(...)`` on the chat_benchmark instances
    (the 3a / 3c seam).

``--resume-mode`` semantics (default ``auto``):
  * ``off``       — pure no-op: NO factory is built, NOTHING is attached, no
    ``resume/`` dir is written. Reproduces today exactly (global invariant #1).
  * ``auto``      — auto-detect prior state under the per-task run dir; a FIRST
    run with no matching state is a pure no-op that only writes the (inert)
    fingerprint/state dir (mirror Harbor's ``is_resuming=False`` branch); a
    second run with identical inputs resumes; a material delta refuses.
  * ``force-fresh`` — wipe any prior state and start fresh.

When ``output_path`` is unset there is no durable run dir to anchor state on, so
resume is impossible — we degrade to a no-op (same as ``off``), preserving the
flag-off invariant.

The fingerprint here is built from the run inputs that are *cheaply available
from ``args`` + the initialized ``lm``* (model repo/revision, decoding params,
seeds, template on/off, num_fewshot, max_model_len, num_samples / pass@k batch
``B``, and a light rendered-config dict). Heavy per-benchmark controlling-file
digests (loaded dataset bytes, grader ``__file__``) are NOT loaded here — they
would require materializing each benchmark's dataset/grader at wiring time; the
decision table is correct without them (they would only ADD refuse-sensitivity).
Each task gets its OWN manager under ``<run_dir>/<task>`` so tasks never share
state, and ``task_name`` is a material field so two tasks never collide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .fingerprint import RunFingerprint, resolve_model_revision

logger = logging.getLogger(__name__)


def _parse_model_args(model_args: Optional[str]) -> dict:
    """Best-effort parse of the ``key=val,key=val`` model_args string into a dict."""
    out: dict = {}
    if not model_args:
        return out
    if isinstance(model_args, dict):
        return dict(model_args)
    for part in str(model_args).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_gen_kwargs(gen_kwargs: Optional[str]) -> dict:
    """Parse the ``--gen_kwargs`` ``temperature=0,top_p=1`` string into a dict.

    Numbers are coerced to int/float so two spellings of the same value hash
    identically; everything else stays a string.
    """
    out: dict = {}
    if not gen_kwargs:
        return out
    for part in str(gen_kwargs).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _world_rank(lm: Any) -> tuple[int, int]:
    return int(getattr(lm, "world_size", 1) or 1), int(getattr(lm, "rank", 0) or 0)


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def build_resume_wiring(args: Any, lm: Any) -> Optional[Any]:
    """Build the per-task ResumeManager factory and stash it on ``args``.

    Returns the factory (also set as ``args.resume_manager_factory``) so the
    caller can ``attach_resume_manager`` to chat benchmark instances, or ``None``
    when resume is disabled (``off`` mode, or no ``output_path``) — in which case
    NOTHING is attached and behavior is byte-identical to today.
    """
    mode = getattr(args, "resume_mode", "auto") or "auto"

    # off -> pure no-op: do not even build a factory (invariant #1).
    if mode == "off":
        args.resume_manager_factory = None
        return None

    output_path = getattr(args, "output_path", None)
    if not output_path:
        # No durable run dir to anchor state -> resume impossible; degrade to no-op.
        logger.info("resume: --resume-mode=%s but no --output_path; resume disabled (no-op).", mode)
        args.resume_manager_factory = None
        return None

    world_size, rank = _world_rank(lm)

    margs = _parse_model_args(getattr(args, "model_args", "") or "")
    model_repo = margs.get("pretrained") or getattr(args, "model_name", None)
    revision = margs.get("revision")
    model_revision = resolve_model_revision(model_repo, revision, allow_network=False)
    # Evalchemy's canonical CLI writes ``max_length``.  Keep the older vLLM
    # spelling for backwards-compatible resume fingerprints.
    max_model_len = margs.get("max_length", margs.get("max_model_len"))
    if max_model_len is not None:
        try:
            max_model_len = int(max_model_len)
        except (TypeError, ValueError):
            max_model_len = None

    gen = _parse_gen_kwargs(getattr(args, "gen_kwargs", None))
    max_tokens = getattr(args, "max_tokens", None)
    decoding: dict = {}
    for k in ("temperature", "top_p", "do_sample"):
        if k in gen:
            decoding[k] = gen[k]
    # max generation length: prefer the explicit --max_tokens, else gen_kwargs alias.
    if max_tokens is not None:
        try:
            decoding["max_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            pass
    elif "max_gen_toks" in gen:
        decoding["max_gen_toks"] = gen["max_gen_toks"]
    num_samples = int(getattr(args, "num_samples", 1) or 1)
    if num_samples > 1:
        decoding["num_samples"] = num_samples

    apply_chat_template = bool(getattr(args, "apply_chat_template", False))
    num_fewshot = getattr(args, "num_fewshot", None)
    passk_batch_size = getattr(args, "passk_batch_size", None)

    seed = getattr(args, "seed", None)
    seed_set = list(seed) if seed is not None else None

    # A light rendered-config dict (material): the resolved knobs that change the
    # run's meaning but are not already covered by the scalars above.
    rendered_config = {
        "annotator_model": getattr(args, "annotator_model", None),
        "limit": getattr(args, "limit", None),
        "predict_only": bool(getattr(args, "predict_only", False)),
        "fewshot_as_multiturn": bool(getattr(args, "fewshot_as_multiturn", False)),
        "system_instruction": getattr(args, "system_instruction", None),
    }

    model_dir = _sanitize(model_repo or "model")
    base_run_dir = Path(output_path) / ".resume" / model_dir

    def factory(task_name: str):
        from .manager import ResumeManager

        fp = RunFingerprint.from_run_inputs(
            model_repo=model_repo,
            model_revision=model_revision,
            task_name=task_name,
            decoding=decoding or None,
            seed_set=seed_set,
            max_model_len=max_model_len,
            num_fewshot=num_fewshot,
            passk_batch_size=passk_batch_size,
            apply_chat_template=apply_chat_template,
            rendered_config=rendered_config,
        )
        run_dir = base_run_dir / _sanitize(task_name)
        return ResumeManager(
            run_dir=run_dir,
            fingerprint=fp,
            mode=mode,
            world_size=world_size,
            rank=rank,
        )

    args.resume_manager_factory = factory
    logger.info(
        "resume: --resume-mode=%s active; per-task state under %s (model=%s, rev=%s).",
        mode,
        base_run_dir,
        model_repo,
        model_revision,
    )
    return factory


def attach_to_chat_benchmarks(task_manager: Any, task_list: list, factory: Any) -> None:
    """Attach a per-task ResumeManager to each chat benchmark instance (3a/3c seam).

    No-op when ``factory`` is ``None`` (off-mode / no output_path) so nothing is
    attached and ``compute`` stays byte-identical to today.
    """
    if factory is None:
        return
    instances = getattr(task_manager, "benchmark_instances", {}) or {}
    for task_name in task_list:
        bench = instances.get(task_name)
        if bench is None:
            continue
        try:
            bench.attach_resume_manager(factory(task_name))
        except Exception as exc:  # pragma: no cover - defensive; never break a run on wiring
            logger.warning("resume: could not attach manager to %s (%s); running without resume.", task_name, exc)
