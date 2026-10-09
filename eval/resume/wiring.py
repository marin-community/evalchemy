"""Construct per-task FineStore resume managers from CLI inputs.

The factory feeds both lm-eval and chat benchmarks. Request state lives in the
FineStore archive, and a fresh evaluation requires a new archive path.

The fingerprint here is built from the run inputs that are *cheaply available
from ``args`` + the initialized ``lm``* (model repo/revision, decoding params,
seeds, template on/off, num_fewshot, max_model_len, num_samples / pass@k batch
``B``, and a light rendered-config dict). Heavy per-benchmark controlling-file
digests (loaded dataset bytes, grader ``__file__``) are NOT loaded here — they
would require materializing each benchmark's dataset/grader at wiring time; the
decision table is correct without them (they would only ADD refuse-sensitivity).
Each task gets its own namespace in the archive, and ``task_name`` is a
material fingerprint field so tasks never collide.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from eval.contracts.finestore_resume import FineStoreResumeManager

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


def build_resume_wiring(args: Any, lm: Any) -> Any:
    """Build the per-task FineStore manager factory and stash it on ``args``.

    Returns the factory (also set as ``args.resume_manager_factory``) so the
    caller can ``attach_resume_manager`` to chat benchmark instances.
    """
    mode = getattr(args, "resume_mode", "auto") or "auto"
    if mode == "off":
        raise ValueError("FineStore resume cannot be disabled")

    finestore_output_path = getattr(args, "finestore_output_path", None)
    if not finestore_output_path:
        raise ValueError("--finestore_output_path is required for resume")

    world_size, rank = _world_rank(lm)

    margs = _parse_model_args(getattr(args, "model_args", "") or "")
    model_repo = margs.get("pretrained") or margs.get("model") or getattr(args, "model_name", None)
    revision = margs.get("revision")
    model_revision = resolve_model_revision(model_repo, revision, allow_network=False)
    # The endpoint may also report this as max_model_len.
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

    def factory(task_name: str):
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
        return FineStoreResumeManager(
            root=finestore_output_path,
            source_prefix=getattr(args, "finestore_output_prefix", "run"),
            task_name=task_name,
            fingerprint=fp,
            mode=mode,
            world_size=world_size,
            rank=rank,
        )

    args.resume_manager_factory = factory
    logger.info(
        "resume: --resume-mode=%s active; per-task state under %s (model=%s, rev=%s).",
        mode,
        finestore_output_path,
        model_repo,
        model_revision,
    )
    return factory


def attach_to_chat_benchmarks(task_manager: Any, task_list: list, factory: Any) -> None:
    """Attach each chat benchmark's FineStore resume manager."""
    instances = getattr(task_manager, "benchmark_instances", {}) or {}
    for task_name in task_list:
        bench = instances.get(task_name)
        if bench is None:
            continue
        bench.attach_resume_manager(factory(task_name))
