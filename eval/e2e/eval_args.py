# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the ``eval.eval`` invocation for an OpenAI-compatible served model.

This mirrors, semantically, Marin's ``build_lm_eval_model_args``
(``lib/marin/src/marin/evaluation/lm_eval.py``) so both projects agree on the
model_args a served endpoint is evaluated with -- but it has **no Marin import**
so the harness core stays installable on evalchemy's base Python (>=3.8) with no
accelerator.

Contract (the source of a class of real bugs, so it is enforced here):

* ``ServedModel.base_url`` is the OpenAI API **root**, i.e. ``.../v1``. It must
  NOT include the adapter path. ``marin-serve`` prints exactly this (``OpenAI:
  {proxy}/v1``).
* :func:`build_model_args` appends the adapter path (``/completions`` or
  ``/chat/completions``) exactly once, tolerating a caller who redundantly
  included it, so the emitted ``base_url`` can never double ``/v1`` or the path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# lm-eval OpenAI-compatible model backends. These are the registry names
# evalchemy resolves via ``lm_eval.api.registry.get_model`` (see eval/eval.py).
LOCAL_COMPLETIONS = "local-completions"
LOCAL_CHAT_COMPLETIONS = "local-chat-completions"

# Adapter -> OpenAI route appended under the /v1 root.
_ADAPTER_PATH = {
    LOCAL_COMPLETIONS: "completions",
    LOCAL_CHAT_COMPLETIONS: "chat/completions",
}

ModelArgValue = Union[str, int, float, bool]


@dataclass(frozen=True)
class ServedModel:
    """A model already served behind an OpenAI-compatible endpoint.

    ``base_url`` is the ``.../v1`` API root. ``api_key`` is sent by lm-eval as
    ``Authorization: Bearer <api_key>`` (used to carry an IAP bearer for the
    Marin controller proxy, or any server that wants a key).
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    tokenizer: Optional[str] = None


def _api_root(base_url: str) -> str:
    """Normalize ``base_url`` to the ``/v1`` root, tolerating an appended adapter path."""
    root = base_url.rstrip("/")
    for path in ("chat/completions", "completions"):
        if root.endswith("/" + path):
            root = root[: -(len(path) + 1)].rstrip("/")
    return root


def adapter_for(apply_chat_template: bool) -> str:
    """Pick the lm-eval adapter: chat endpoint when a chat template is applied."""
    return LOCAL_CHAT_COMPLETIONS if apply_chat_template else LOCAL_COMPLETIONS


def endpoint_url(base_url: str, adapter: str) -> str:
    """Return the concrete OpenAI endpoint URL for ``adapter`` under the API root."""
    try:
        path = _ADAPTER_PATH[adapter]
    except KeyError:
        raise ValueError(f"unknown adapter {adapter!r}; expected one of {sorted(_ADAPTER_PATH)}")
    return _api_root(base_url) + "/" + path


def _format_value(value: ModelArgValue) -> str:
    # lm-eval parses model_args as comma-separated key=value; a value cannot
    # contain a comma or the pair boundary is lost.
    text = "True" if value is True else "False" if value is False else str(value)
    if "," in text:
        raise ValueError(f"model_args value cannot contain ',': {text!r}")
    return text


def _format_key(key: str) -> str:
    if not key or "," in key or "=" in key:
        raise ValueError(f"invalid model_args key: {key!r}")
    return key


def build_model_args(
    served: ServedModel,
    adapter: str,
    extra: Optional[Dict[str, ModelArgValue]] = None,
) -> str:
    """Build the comma-delimited ``--model_args`` string for a served model."""
    args: "Dict[str, ModelArgValue]" = {
        "model": served.model,
        "base_url": endpoint_url(served.base_url, adapter),
        # The served model already tokenizes; keep requests as text and let
        # lm-eval use a HF tokenizer only for length bookkeeping.
        "tokenizer_backend": "huggingface",
        "tokenized_requests": False,
    }
    if served.api_key is not None:
        args["api_key"] = served.api_key
    if served.tokenizer is not None:
        args["tokenizer"] = served.tokenizer
    if extra:
        args.update(extra)
    return ",".join(f"{_format_key(k)}={_format_value(v)}" for k, v in args.items())


@dataclass(frozen=True)
class EvalInvocation:
    """Everything needed to run one ``eval.eval`` against a served model."""

    served: ServedModel
    tasks: List[str]
    output_path: str
    apply_chat_template: bool = False
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    batch_size: Union[int, str] = 1
    seed: Optional[int] = 1234
    gen_kwargs: Optional[str] = None
    extra_model_args: Dict[str, ModelArgValue] = field(default_factory=dict)
    # Extra ``eval.eval`` flags forwarded verbatim (e.g. ``--num_samples 8
    # --pass_at_k 1,8``). Appended last so lm-eval's argparse lets them override
    # our defaults; this is how pass@k / --predict_only / future flags reach the
    # eval without re-plumbing each one here.
    extra_args: List[str] = field(default_factory=list)

    @property
    def adapter(self) -> str:
        return adapter_for(self.apply_chat_template)

    def model_args(self) -> str:
        return build_model_args(self.served, self.adapter, self.extra_model_args)


def build_eval_argv(inv: EvalInvocation, python: str = "python") -> List[str]:
    """Build the argv for ``python -m eval.eval`` for this invocation.

    Uses the **bare** ``--apply_chat_template`` flag. That parser option is
    ``nargs="?", const=True`` (eval/lm_eval_compat.py): passing a value like
    ``True`` would be read as a chat-*template name*, not the boolean -- a real
    footgun this function exists to avoid.
    """
    if not inv.tasks:
        raise ValueError("EvalInvocation.tasks must be non-empty")

    argv: List[str] = [
        python,
        "-m",
        "eval.eval",
        "--model",
        inv.adapter,
        "--model_args",
        inv.model_args(),
        "--tasks",
        ",".join(inv.tasks),
        "--output_path",
        inv.output_path,
        "--log_samples",
    ]
    if inv.apply_chat_template:
        argv.append("--apply_chat_template")  # bare flag => const=True
    if inv.limit is not None:
        argv += ["--limit", str(inv.limit)]
    if inv.num_fewshot is not None:
        argv += ["--num_fewshot", str(inv.num_fewshot)]
    if inv.batch_size is not None:
        argv += ["--batch_size", str(inv.batch_size)]
    if inv.seed is not None:
        # evalchemy/lm-eval accept a 4-tuple "python,numpy,torch,fewshot"; a
        # single value sets all four (see eval/eval.py seed handling).
        argv += ["--seed", str(inv.seed)]
    if inv.gen_kwargs:
        argv += ["--gen_kwargs", inv.gen_kwargs]
    argv += list(inv.extra_args)  # verbatim eval.eval passthrough (last => can override)
    return argv
