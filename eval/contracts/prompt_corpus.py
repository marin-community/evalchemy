"""Render one benchmark's prompt corpus without contacting a model.

A benchmark builds its prompts inside ``generate_responses``, so the only way to
see what it will actually send is to run that method. This module drives it with
an LM stand-in that records each rendered request and returns empty completions,
which makes the longest prompt in a benchmark measurable offline and ahead of any
run. ``scripts/benchmarks/compute_prompt_lengths.py`` uses it to write
``eval/contracts/prompt_lengths.json``; the contract test uses it to detect a
stale entry.

Prompt text depends on the chat template and the few-shot configuration, and
token counts depend on the tokenizer, so every measurement is taken with one
declared reference tokenizer (:data:`REFERENCE_TOKENIZER`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from eval.contracts.prompt_length import REFERENCE_TOKENIZER, BenchmarkPromptLength

PLACEHOLDER_CREDENTIAL_VARIABLES = ("OPENAI_API_KEY", "JUDGE_API_KEY")
PLACEHOLDER_CREDENTIAL = "prompt-corpus-render"


class PromptCaptureLM:
    """LM stand-in that records each rendered prompt instead of generating.

    Exposes the surface the benchmarks read off a model: the distributed
    coordinates, the chat template, and the tokenizer/model identity some
    benchmarks use to size few-shot prompts.
    """

    rank = 0
    world_size = 1

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pretrained: str = REFERENCE_TOKENIZER):
        self.tokenizer = tokenizer
        # Benchmarks read the model identity under several names when they size a
        # prompt or pick a conversation template; all of them mean the reference.
        self.pretrained = pretrained
        self.model = pretrained
        self.model_identifier = pretrained
        self.prompts: list[Any] = []

    def apply_chat_template(self, messages: Sequence[Mapping[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_until(self, instances: Sequence[Any]) -> list[str]:
        self.prompts.extend(instance.args[0] for instance in instances)
        return ["" for _ in instances]


def load_reference_tokenizer(name: str = REFERENCE_TOKENIZER) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name)


@contextlib.contextmanager
def placeholder_credentials():
    """Satisfy benchmarks that refuse to construct without a judge credential.

    Nothing is sent: the LM stand-in never reaches an endpoint and no judge is
    called. Credentials already in the environment are left alone, and anything
    set here is removed again on the way out.
    """
    previous = {name: os.environ.get(name) for name in PLACEHOLDER_CREDENTIAL_VARIABLES}
    for name, value in previous.items():
        if not value:
            os.environ[name] = PLACEHOLDER_CREDENTIAL
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def render_prompt_corpus(task_name: str, tokenizer: PreTrainedTokenizerBase) -> BenchmarkPromptLength:
    """Measure every distinct prompt ``task_name`` renders, with no model call."""
    # Local to break the cycle: eval.task reads the stored lengths from
    # eval.contracts.prompt_length, which this module imports.
    from eval.task import TaskManager

    with placeholder_credentials():
        manager = TaskManager(task_list=[task_name])
        failure = manager.load_failures.get(task_name)
        if failure is not None:
            raise failure
        benchmark = manager.get_benchmark(task_name)
        model = PromptCaptureLM(tokenizer)
        generation_result = benchmark.generate_responses(model)
    _release(generation_result)
    return measure_prompts(task_name, model.prompts, tokenizer)


def measure_prompts(
    task_name: str, prompts: Sequence[Any], tokenizer: PreTrainedTokenizerBase
) -> BenchmarkPromptLength:
    """Reduce captured prompts to the stored measurements."""
    if not prompts:
        raise ValueError(f"{task_name} rendered no prompts")
    rendered = sorted({_prompt_text(prompt, tokenizer) for prompt in prompts})
    digest = hashlib.sha256(json.dumps(rendered, ensure_ascii=False).encode("utf-8")).hexdigest()
    longest = max(rendered, key=len)
    return BenchmarkPromptLength(
        distinct_prompt_count=len(rendered),
        longest_prompt_chars=len(longest),
        longest_prompt_tokens=max(len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered),
        sha256=digest,
    )


def _prompt_text(prompt: Any, tokenizer: PreTrainedTokenizerBase) -> str:
    """Return one captured request as the text the endpoint will receive."""
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "prompt"):
        return str(prompt.prompt)
    if isinstance(prompt, Sequence) and all(isinstance(message, Mapping) for message in prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"unsupported rendered prompt: {type(prompt).__name__}")


def _release(generation_result: Any) -> None:
    """Drop the temporary grader inputs a generation pass may have created."""
    if not isinstance(generation_result, Mapping):
        return
    for key in ("temp_dir_obj", "artifacts"):
        owner = generation_result.get(key)
        if owner is not None and hasattr(owner, "cleanup"):
            owner.cleanup()
