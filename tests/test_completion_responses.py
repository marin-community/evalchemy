"""Regression coverage for OpenAI-compatible reasoning responses."""

import asyncio
import json
from collections import Counter

import pytest
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion

from eval import robust_api  # noqa: F401 - installs the lm-eval adapter patch
from eval.completion_response import (
    CompletionClassification,
    CompletionContentPolicy,
    CompletionText,
    FailedGeneration,
    completion_response_from_chat_choice,
)
from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import record_sample_metrics
from eval.robust_api import capture_endpoint_failures
from eval.sample_logging import canonicalize_samples
from eval.task import BaseBenchmark


def _adapter(policy: CompletionContentPolicy = CompletionContentPolicy.COMBINE):
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = policy
    adapter.completion_responses = []
    return adapter


def _openai_adapter(model: str) -> OpenAIChatCompletion:
    adapter = object.__new__(OpenAIChatCompletion)
    adapter.model = model
    adapter._max_gen_toks = 256
    return adapter


def _openai_payload(model: str) -> dict:
    return _openai_adapter(model)._create_payload(
        messages=[{"role": "user", "content": "Question"}],
        gen_kwargs={
            "do_sample": False,
            "max_gen_toks": 32,
            "temperature": 0,
            "until": ["<|im_end|>", "\nQuestion:"],
        },
    )


def _local_chat_payload(stops: list[str]) -> dict:
    adapter = object.__new__(LocalChatCompletion)
    adapter.model = "served-model"
    adapter._max_gen_toks = 256
    return adapter._create_payload(
        messages=[{"role": "user", "content": "Question"}],
        gen_kwargs={"max_gen_toks": 32, "until": stops},
    )


def _local_completions_payload(stops: list[str]) -> dict:
    adapter = object.__new__(LocalCompletionsAPI)
    adapter.model = "served-model"
    adapter._max_gen_toks = 256
    return adapter._create_payload(
        messages="Question",
        generate=True,
        gen_kwargs={"max_gen_toks": 32, "until": stops},
    )


def test_gpt5_payload_uses_openai_provider_generation_controls():
    payload = _openai_payload("gpt-5")

    assert "stop" not in payload
    assert payload["temperature"] == 1


@pytest.mark.parametrize("model", ["qwen3.5-9b", "provider-model-5"])
def test_non_openai_alias_with_five_retains_configured_generation_controls(model):
    payload = _openai_payload(model)

    assert payload["stop"][:2] == ["\nQuestion:", "<|im_end|>"]
    assert payload["temperature"] == 0


def test_local_chat_payload_sends_only_token_sentinels():
    payload = _local_chat_payload(
        [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "Question:",
            "\nQ:",
            "\n[Question]",
            "\nUser:",
        ]
    )

    assert payload["stop"] == ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]


def test_local_chat_payload_without_sentinels_ends_the_turn_at_eos():
    payload = _local_chat_payload(["Question:", "\nQ:", "\nUser:", "\nAssistant:"])

    assert "stop" not in payload


def test_local_completions_payload_uses_the_same_bounded_stop_policy():
    stops = ["<|im_end|>", "<|eot_id|>", "Question:", "\nQ:", "\n[Question]", "\nUser:"]

    assert _local_completions_payload(stops)["stop"] == ["Question:", "\nQ:", "\n[Question]", "\nUser:"]


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


def test_failed_completion_is_empty_and_classified_in_sample_artifact():
    response = FailedGeneration("model_transport")
    record = canonicalize_samples(
        "task",
        [{"resps": [[response]], "metrics": ["accuracy"], "accuracy": 0.0}],
    )[0]

    assert response == ""
    assert record["failure_category"] == "model_transport"
    assert record["completion_responses"][0][0]["normalized_content"] == ""


def test_one_failed_async_request_returns_an_empty_classified_response():
    adapter = object.__new__(LocalChatCompletion)
    adapter._concurrent = 2
    adapter.verify_certificate = True
    adapter.timeout = 1
    adapter.max_retries = 1
    adapter._batch_size = 1
    adapter.tokenizer = None
    adapter.max_length = None

    async def fake_model_call(*, messages, **_kwargs):
        if messages[0] == "bad":
            raise TimeoutError("endpoint timed out")
        return ["ok"]

    adapter.amodel_call = fake_model_call

    async def fetch():
        with capture_endpoint_failures() as failures:
            outputs = await adapter.get_batched_requests(["bad", "good"], ["a", "b"], gen_kwargs={})
        return outputs, failures

    outputs, failures = asyncio.run(fetch())

    assert isinstance(outputs[0][0], FailedGeneration)
    assert outputs[0][0] == ""
    assert outputs[1] == ["ok"]
    assert failures.counts == {FailureCategory.MODEL_TRANSPORT: 1}


def test_generation_retry_loop_is_bounded_by_one_agent_timeout():
    adapter = object.__new__(LocalChatCompletion)
    adapter._concurrent = 1
    adapter.verify_certificate = True
    adapter.timeout = 0.01
    adapter.max_retries = 8
    adapter._batch_size = 1
    adapter.tokenizer = None
    adapter.max_length = None
    never_returns = asyncio.Event()

    async def fake_model_call(*, messages, **_kwargs):
        if messages[0] == "slow":
            await never_returns.wait()
        return ["ok"]

    adapter.amodel_call = fake_model_call

    async def fetch():
        with capture_endpoint_failures() as failures:
            outputs = await adapter.get_batched_requests(
                ["slow", "queued"],
                ["slow-cache", "queued-cache"],
                gen_kwargs={},
            )
        return outputs, failures

    outputs, failures = asyncio.run(fetch())
    record = canonicalize_samples(
        "task",
        [{"resps": [outputs[0]], "metrics": ["accuracy"], "accuracy": 0.0}],
    )[0]

    assert outputs[0] == [""]
    assert outputs[1] == ["ok"]
    assert failures.counts == {FailureCategory.AGENT_TIMEOUT: 1}
    assert record["failure_category"] == "AgentTimeoutError"


def test_loglikelihood_request_error_keeps_session_open_until_siblings_settle():
    adapter = object.__new__(LocalChatCompletion)
    adapter._concurrent = 2
    adapter.verify_certificate = True
    adapter.timeout = 1
    adapter.max_retries = 1
    adapter._batch_size = 1
    adapter.tokenizer = None
    adapter.max_length = None
    session_states = []

    async def fake_model_call(*, session, messages, **_kwargs):
        if messages[0] == "bad":
            raise RuntimeError("serve error")
        await asyncio.sleep(0.01)
        session_states.append(session.closed)
        return [(-1.0, False)]

    adapter.amodel_call = fake_model_call

    async def fetch():
        with pytest.raises(RuntimeError, match="serve error"):
            await adapter.get_batched_requests(
                ["bad", "good"],
                ["a", "b"],
                generate=False,
            )
        # The old gather path left the sibling alive after closing its session.
        await asyncio.sleep(0.02)

    asyncio.run(fetch())

    assert session_states == [False]


def test_chat_request_without_client_tokenizer_skips_preflight_and_sends_cap():
    adapter = object.__new__(LocalChatCompletion)
    adapter._concurrent = 2
    adapter.verify_certificate = True
    adapter.timeout = 1
    adapter.max_retries = 1
    adapter._batch_size = 1
    adapter.tokenizer = None
    adapter.max_length = 4095
    sent_kwargs = []

    async def fake_model_call(*, gen_kwargs, **_kwargs):
        sent_kwargs.append(gen_kwargs)
        return ["ok"]

    adapter.amodel_call = fake_model_call
    messages = [{"role": "user", "content": "question"}]

    async def fetch():
        return await adapter.get_batched_requests([messages], ["cache"], gen_kwargs={"max_gen_toks": 128})

    outputs = asyncio.run(fetch())

    assert outputs == [["ok"]]
    assert sent_kwargs == [{"max_gen_toks": 128}]


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
        samples = [{"resps": [[generated]], "metrics": ["accuracy"], "accuracy": 1.0}]
    else:
        example = {"problem": "1 + 1", "answer": "2", "model_output": generated}
        record_sample_metrics(example, accuracy=1.0)
        samples = _NativeBenchmark().to_samples({"examples": [example]}, {})
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
        samples = [{"resps": [[generated]], "metrics": ["accuracy"], "accuracy": 1.0}]
    else:
        example = {"problem": "1 + 1", "answer": "2", "model_output": generated}
        record_sample_metrics(example, accuracy=1.0)
        samples = _NativeBenchmark().to_samples({"examples": [example]}, {})
    record = json.loads(json.dumps(canonicalize_samples(pipeline, samples)[0]))

    expected_scorer_text = expected if policy == CompletionContentPolicy.COMBINE else message.get("content") or ""
    assert str(generated) == expected_scorer_text
    assert record["completion_responses"][0][0]["classification"] == classification
    assert record["completion_responses"][0][0]["raw_choice"] == choice
    assert record["completion_responses"][0][0]["usage"] == {"completion_tokens": 4}
    assert record["completion_responses"][0][0]["content_policy"] == policy
