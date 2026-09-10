from types import SimpleNamespace

import pytest

from eval.chat_benchmarks.FinanceBench import judge as finance_judge
from eval.task import TaskManager


class _FakeAsyncOpenAI:
    constructor_kwargs = None
    request_kwargs = None

    def __init__(self, **kwargs):
        type(self).constructor_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        type(self).request_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))])


class _FailingAsyncOpenAI(_FakeAsyncOpenAI):
    async def create(self, **kwargs):
        raise RuntimeError("judge unavailable")


def test_financebench_judge_uses_dedicated_endpoint_and_key(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FakeAsyncOpenAI)

    judgments = finance_judge.judge(
        [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
        "judge-model",
        api_key="judge-key",
        base_url="https://judge.example/v1",
    )

    assert judgments == [("correct", "correct")]
    assert _FakeAsyncOpenAI.constructor_kwargs["api_key"] == "judge-key"
    assert _FakeAsyncOpenAI.constructor_kwargs["base_url"] == "https://judge.example/v1"
    assert _FakeAsyncOpenAI.request_kwargs["model"] == "judge-model"


def test_financebench_judge_api_failure_propagates(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FailingAsyncOpenAI)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        finance_judge.judge(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )


def test_financebench_malformed_judge_response_propagates(monkeypatch):
    class MalformedAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="maybe"))])

    monkeypatch.setattr(finance_judge, "AsyncOpenAI", MalformedAsyncOpenAI)

    with pytest.raises(ValueError, match="unrecognized FinanceBench judgment"):
        finance_judge.judge(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )


def test_financebench_does_not_accept_candidate_endpoint_key_as_judge_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "candidate-endpoint-key")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    manager = TaskManager(task_list=["FinanceBench"])

    assert manager.get_benchmark("FinanceBench") is None
    assert "JUDGE_API_KEY" in str(manager.load_failures["FinanceBench"])
