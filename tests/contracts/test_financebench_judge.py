import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.contracts.task_outcome import validate_result_document
from eval.eval import CHAT_BENCHMARK_ROUTE, evaluate
from eval.graders import answer_equivalence
from eval.limits import DEFAULT_CONTEXT_SAFETY_TOKENS
from eval.task import TaskManager


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))])


class _FailingAsyncOpenAI(_FakeAsyncOpenAI):
    async def create(self, **kwargs):
        raise RuntimeError("judge unavailable")


class _MixedJudgeAsyncOpenAI(_FakeAsyncOpenAI):
    async def create(self, **kwargs):
        if "Question:\nUnavailable?" in kwargs["messages"][0]["content"]:
            raise TimeoutError("judge unavailable")
        return await super().create(**kwargs)


class _CandidateModel:
    rank = 0
    world_size = 1

    def __init__(self):
        self.generated_ids = []
        self.request_kwargs = []

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, instances):
        self.generated_ids.extend(instance.idx for instance in instances)
        self.request_kwargs.extend(dict(instance.args[1]) for instance in instances)
        return ["candidate answer" for _ in instances]


def test_financebench_judge_failure_is_saved_per_trial_without_losing_other_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _MixedJudgeAsyncOpenAI)
    rows = [
        {"question": "Available?", "answer": "yes", "evidence_text": "Evidence"},
        {"question": "Unavailable?", "answer": "no", "evidence_text": "Evidence"},
    ]
    data_file = tmp_path / "financebench.jsonl"
    data_file.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    benchmark = FinanceBenchBenchmark(data_file=str(data_file), judge_api_key="judge-key")
    model = _CandidateModel()

    result = evaluate(
        lm=model,
        task_manager=SimpleNamespace(tasks={"FinanceBench": benchmark}, get_benchmark=lambda _task: benchmark),
        pretrain_task_manager=SimpleNamespace(all_tasks={}),
        task_list=["FinanceBench"],
        task_routes={"FinanceBench": CHAT_BENCHMARK_ROUTE},
        batch_sizes_list=[1],
        args=Namespace(model="local-chat-completions", log_samples=True),
    )

    assert result["task_outcomes"]["FinanceBench"]["failure"] is None
    score = result["results"]["FinanceBench"]
    samples = result["samples"]["FinanceBench"]
    assert model.generated_ids == [0, 1]
    assert score["accuracy"] == 1.0
    assert score["num_judged"] == 1
    assert score["num_judge_failed"] == 1
    assert samples[0]["accuracy"] == 1.0
    assert "accuracy" not in samples[1]
    assert samples[1]["failure_category"] == "grader_infrastructure"
    assert samples[1]["judge_error"]["category"] == "grader_infrastructure"
    assert samples[1]["judge_error"]["exception_type"] == "TimeoutError"
    assert "judge_error" not in samples[1]["doc"]
    assert result["task_outcomes"]["FinanceBench"]["failure_counts"] == {"grader_infrastructure": 1}
    validate_result_document(result)


def test_financebench_all_judge_failures_return_no_accuracy(monkeypatch):
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _FailingAsyncOpenAI)
    benchmark = FinanceBenchBenchmark(judge_api_key="judge-key")
    generated = {
        "examples": [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
        "judge_model": "judge-model",
    }

    scored = benchmark.evaluate_responses(generated)

    assert scored["accuracy"] is None
    assert scored["num_judged"] == 0
    assert scored["num_judge_failed"] == 1


def test_financebench_does_not_accept_candidate_endpoint_key_as_judge_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "candidate-endpoint-key")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    manager = TaskManager(task_list=["FinanceBench"])

    assert manager.get_benchmark("FinanceBench") is None
    assert "JUDGE_API_KEY" in str(manager.load_failures["FinanceBench"])


def test_financebench_auto_annotator_uses_dedicated_judge_model(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-oss-120b")

    manager = TaskManager(task_list=["FinanceBench"], annotator_model="auto")

    assert manager.load_failures == {}
    assert manager.get_benchmark("FinanceBench").judge_model == "openai/gpt-oss-120b"


def test_financebench_sample_cap_limits_generated_and_returned_examples(tmp_path):
    rows = [{"question": f"Question {index}", "answer": str(index), "evidence_text": "Evidence"} for index in range(3)]
    data_file = tmp_path / "financebench.jsonl"
    data_file.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    benchmark = FinanceBenchBenchmark(data_file=str(data_file), judge_api_key="judge-key")
    benchmark.set_evaluation_limits(limit=1)
    model = _CandidateModel()

    generated = benchmark.generate_responses(model)

    assert model.generated_ids == [0]
    assert [example["question"] for example in generated["examples"]] == ["Question 0"]


@pytest.mark.parametrize("requested_max_tokens", [73, None])
def test_financebench_request_obeys_configured_context_output_and_sample_limits(monkeypatch, requested_max_tokens):
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    manager = TaskManager(
        task_list=["FinanceBench"],
        max_length=32768,
        max_tokens=requested_max_tokens,
        limit=1,
    )
    assert manager.load_failures == {}
    benchmark = manager.get_benchmark("FinanceBench")
    model = _CandidateModel()

    generated = benchmark.generate_responses(model)

    expected_cap = requested_max_tokens
    if expected_cap is None:
        expected_cap = 32768 - manager.prompt_lengths.max_prompt_tokens("FinanceBench") - DEFAULT_CONTEXT_SAFETY_TOKENS
    assert model.generated_ids == [0]
    assert len(generated["examples"]) == 1
    assert model.request_kwargs[0]["max_new_tokens"] == expected_cap
    if requested_max_tokens is None:
        assert expected_cap > 4096
