"""Regression coverage for OpenAI-compatible reasoning responses."""

from collections import Counter
import json

import pytest

from eval import robust_api  # noqa: F401 - installs the lm-eval adapter patch
from eval.completion_response import (
    CompletionClassification,
    CompletionContentPolicy,
    CompletionText,
    completion_response_from_chat_choice,
)
from eval.sample_logging import canonicalize_samples
from eval.task import BaseBenchmark
from lm_eval.models.openai_completions import LocalChatCompletion


def _adapter(policy: CompletionContentPolicy = CompletionContentPolicy.COMBINE):
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = policy
    adapter.completion_responses = []
    return adapter


class _NativeBenchmark(BaseBenchmark):
    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError


def test_local_chat_completion_preserves_reasoning_when_final_content_is_null():
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "2 + 2 = 4"},
            }
        ]
    }

    generated = _adapter().parse_generations(response)

    assert generated == ["2 + 2 = 4"]
    assert isinstance(generated[0], CompletionText)


def test_successful_empty_chat_responses_invalidate_result_quality():
    responses = [
        completion_response_from_chat_choice(
            {"id": f"chatcmpl-{index}", "usage": {"completion_tokens": 2}},
            {"index": 0, "finish_reason": "stop", "message": {"content": None}},
        )
        for index in range(3)
    ]

    classifications = Counter(response.classification for response in responses)

    assert robust_api.completion_response_quality_invalid(classifications)


@pytest.mark.parametrize(
    "choice",
    [
        {"index": 0, "finish_reason": "stop", "message": {"content": None, "reasoning_content": "reasoning"}},
        {"index": 0, "finish_reason": "stop", "message": {"content": None, "reasoning": "reasoning"}},
        {"index": 0, "finish_reason": "stop", "message": {"content": None}, "reasoning_content": "reasoning"},
        {"index": 0, "finish_reason": "stop", "message": {"content": None}, "reasoning": "reasoning"},
    ],
)
@pytest.mark.parametrize("pipeline", ["lm_eval_native", "evalchemy_native"])
def test_reasoning_aliases_are_scored_and_audited_in_every_task_path(choice, pipeline):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}, "choices": [choice]}
    generated = _adapter().parse_generations(response)[0]

    if pipeline == "lm_eval_native":
        samples = [{"resps": [[generated]]}]
    else:
        samples = _NativeBenchmark().to_samples(
            {"examples": [{"problem": "1 + 1", "answer": "2", "model_output": generated}]}, {}
        )
    record = json.loads(json.dumps(canonicalize_samples(pipeline, samples)[0]))

    assert str(generated) == "reasoning"
    assert record["completion_responses"][0][0]["reasoning_content"] == "reasoning"
    assert record["completion_responses"][0][0]["classification"] == CompletionClassification.REASONING_ONLY
    assert record["completion_responses"][0][0]["raw_choice"] == choice


def test_vllm_qwen_message_reasoning_is_not_scored_as_empty():
    response = {
        "id": "chatcmpl-vllm-qwen",
        "usage": {"completion_tokens": 512},
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "9 eggs times $2 per egg equals 18"},
            }
        ],
    }

    generated = _adapter().parse_generations(response)[0]

    assert str(generated) == "9 eggs times $2 per egg equals 18"
    assert generated.response.classification == CompletionClassification.REASONING_ONLY_TRUNCATED


@pytest.mark.parametrize(
    ("message", "finish_reason", "expected", "classification"),
    [
        ({"content": "final"}, "stop", "final", CompletionClassification.FINAL),
        (
            {"content": "final", "reasoning_content": "reasoning"},
            "stop",
            "reasoning\n\nfinal",
            CompletionClassification.REASONING_AND_FINAL,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "length",
            "reasoning",
            CompletionClassification.REASONING_ONLY_TRUNCATED,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "stop",
            "reasoning",
            CompletionClassification.REASONING_ONLY,
        ),
        ({"content": None, "reasoning_content": None}, "stop", "", CompletionClassification.EMPTY),
    ],
)
def test_completion_response_normalizes_every_content_shape(message, finish_reason, expected, classification):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}}
    choice = {"index": 0, "finish_reason": finish_reason, "message": message}

    completion = completion_response_from_chat_choice(response, choice)

    assert completion.normalized_content() == expected
    assert completion.classification == classification
    assert completion.artifact()["raw_choice"] == choice
    assert completion.artifact()["usage"] == {"completion_tokens": 4}
    assert completion.artifact()["provider_metadata"] == {"id": "completion-1"}


@pytest.mark.parametrize("pipeline", ["lm_eval_native", "evalchemy_native"])
@pytest.mark.parametrize("policy", [CompletionContentPolicy.COMBINE, CompletionContentPolicy.FINAL_ONLY])
@pytest.mark.parametrize(
    ("message", "finish_reason", "expected", "classification"),
    [
        ({"content": "final"}, "stop", "final", CompletionClassification.FINAL),
        (
            {"content": "final", "reasoning_content": "reasoning"},
            "stop",
            "reasoning\n\nfinal",
            CompletionClassification.REASONING_AND_FINAL,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "length",
            "reasoning",
            CompletionClassification.REASONING_ONLY_TRUNCATED,
        ),
        (
            {"content": None, "reasoning_content": "reasoning"},
            "stop",
            "reasoning",
            CompletionClassification.REASONING_ONLY,
        ),
        ({"content": None, "reasoning_content": None}, "stop", "", CompletionClassification.EMPTY),
    ],
)
def test_reasoning_responses_are_scored_and_audited_in_every_task_path(
    pipeline, policy, message, finish_reason, expected, classification
):
    response = {"id": "completion-1", "usage": {"completion_tokens": 4}}
    choice = {"index": 0, "finish_reason": finish_reason, "message": message}
    generated = _adapter(policy).parse_generations({**response, "choices": [choice]})[0]

    if pipeline == "lm_eval_native":
        samples = [{"resps": [[generated]]}]
    else:
        samples = _NativeBenchmark().to_samples(
            {"examples": [{"problem": "1 + 1", "answer": "2", "model_output": generated}]}, {}
        )
    record = json.loads(json.dumps(canonicalize_samples(pipeline, samples)[0]))

    expected_scorer_text = expected if policy == CompletionContentPolicy.COMBINE else message.get("content") or ""
    assert str(generated) == expected_scorer_text
    assert record["completion_responses"][0][0]["classification"] == classification
    assert record["completion_responses"][0][0]["raw_choice"] == choice
    assert record["completion_responses"][0][0]["usage"] == {"completion_tokens": 4}
    assert record["completion_responses"][0][0]["content_policy"] == policy
