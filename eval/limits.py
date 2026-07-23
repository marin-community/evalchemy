"""One resolved evaluation-limit contract for every Evalchemy execution path.

``lm-eval`` calls its limits ``max_length`` (model/context window) and
``max_gen_toks`` (generation).  Evalchemy's chat benchmarks historically used
``max_new_tokens`` while OpenAI-compatible endpoints use ``max_tokens``.  The
different spellings are adapter details, not independent user controls.

This module is deliberately dependency-free: resolve the two user-facing
limits once at CLI ingress, materialize the backend-specific representations,
then pass the same resolved values to native and custom benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

_MAX_OUTPUT_ALIASES = ("max_tokens", "max_new_tokens", "max_gen_toks")
_MODEL_LENGTH_ALIASES = ("max_length", "max_model_len")


def parse_key_value_args(value: Optional[str | Mapping[str, Any]]) -> dict[str, Any]:
    """Parse lm-eval's comma-separated ``key=value`` notation without coercion."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    parsed: dict[str, Any] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value in argument string, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in argument string item {item!r}")
        parsed[key] = raw_value.strip()
    return parsed


def format_key_value_args(values: Mapping[str, Any]) -> str:
    """Render a deterministic lm-eval ``key=value`` argument string."""
    return ",".join(f"{key}={value}" for key, value in values.items())


def _as_positive_int(value: Any, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{source} must be a positive integer, got {parsed}")
    return parsed


def _resolve_one(name: str, values: Iterable[tuple[str, Any]]) -> Optional[int]:
    seen: list[tuple[str, int]] = []
    for source, value in values:
        if value is not None:
            seen.append((source, _as_positive_int(value, source)))
    if not seen:
        return None
    distinct = {value for _, value in seen}
    if len(distinct) != 1:
        rendered = ", ".join(f"{source}={value}" for source, value in seen)
        raise ValueError(f"conflicting {name} values: {rendered}; supply one consistent limit")
    return seen[0][1]


@dataclass(frozen=True)
class EvaluationLimits:
    """Resolved model-context and model-output limits for one evaluation run."""

    max_length: Optional[int] = None
    max_tokens: Optional[int] = None

    def apply(self, args: Any) -> None:
        """Materialize this contract in the representations each adapter consumes.

        ``model_args.max_length`` reaches every lm-eval model adapter.  Native
        lm-eval tasks consume ``gen_kwargs.max_gen_toks``; custom benchmarks
        consume ``args.max_tokens`` and are additionally protected by
        :class:`eval.task.BaseBenchmark` at request construction time.
        """
        model_args = parse_key_value_args(getattr(args, "model_args", None))
        gen_kwargs = parse_key_value_args(getattr(args, "gen_kwargs", None))

        if self.max_length is not None:
            model_args["max_length"] = self.max_length
            # vLLM's LM adapter historically calls this spelling.  Keep it in
            # sync only when a legacy caller already selected that adapter key.
            if "max_model_len" in model_args:
                model_args["max_model_len"] = self.max_length

        if self.max_tokens is not None:
            for alias in _MAX_OUTPUT_ALIASES:
                gen_kwargs.pop(alias, None)
            # lm-eval's native task API expects this spelling.  The custom
            # benchmark layer converts it to its endpoint-specific spelling.
            gen_kwargs["max_gen_toks"] = self.max_tokens

        args.model_args = format_key_value_args(model_args)
        args.gen_kwargs = format_key_value_args(gen_kwargs) if gen_kwargs else None
        args.max_length = self.max_length
        args.max_tokens = self.max_tokens


def resolve_evaluation_limits(args: Any) -> EvaluationLimits:
    """Resolve and materialize Evalchemy's sole max-length/max-output contract.

    ``--max_length`` and ``--max_tokens`` are the canonical public inputs.
    Existing ``model_args`` / ``gen_kwargs`` spellings remain supported as
    compatibility aliases, but disagreement is a hard error rather than a
    benchmark-path-dependent result.
    """
    model_args = parse_key_value_args(getattr(args, "model_args", None))
    gen_kwargs = parse_key_value_args(getattr(args, "gen_kwargs", None))

    max_length = _resolve_one(
        "max_length",
        [("--max_length", getattr(args, "max_length", None))]
        + [(f"model_args.{alias}", model_args.get(alias)) for alias in _MODEL_LENGTH_ALIASES],
    )
    max_tokens = _resolve_one(
        "max_tokens",
        [("--max_tokens", getattr(args, "max_tokens", None))]
        + [(f"gen_kwargs.{alias}", gen_kwargs.get(alias)) for alias in _MAX_OUTPUT_ALIASES],
    )
    limits = EvaluationLimits(max_length=max_length, max_tokens=max_tokens)
    limits.apply(args)
    return limits
