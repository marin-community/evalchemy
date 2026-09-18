import asyncio
import json
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from eval.chat_benchmarks.FinanceBench import judge as finance_judge
from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.contracts.prompt_length import resolve_task_max_tokens
from eval.contracts.task_outcome import validate_result_document
from eval.eval import CHAT_BENCHMARK_ROUTE, evaluate
from eval.task import TaskManager


class _FakeAsyncOpenAI:
    constructor_kwargs = None
    request_kwargs = None

    def __init__(self, **kwargs):
        type(self).constructor_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def create(self, **kwargs):
        type(self).request_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))])


class _FailingAsyncOpenAI(_FakeAsyncOpenAI):
    async def create(self, **kwargs):
        raise RuntimeError("judge unavailable")


class _ReasoningJudgeAsyncOpenAI(_FakeAsyncOpenAI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = 0

    async def create(self, **kwargs):
        self.requests += 1
        if self.requests == 1:
            choice = SimpleNamespace(message=SimpleNamespace(content=""))
        elif self.requests == 2:
            choice = SimpleNamespace(message=SimpleNamespace(content=""))
        else:
            choice = SimpleNamespace(message=SimpleNamespace(content="correct"))
        return SimpleNamespace(choices=[choice])


class _MixedJudgeAsyncOpenAI(_FakeAsyncOpenAI):
    async def create(self, **kwargs):
        if "Question: Unavailable?" in kwargs["messages"][0]["content"]:
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


@pytest.fixture
def judge_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            if body["model"] == "retry-model" and len(requests) <= 3:
                payload = b'{"error":{"message":"judge temporarily unavailable","type":"server_error"}}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = json.dumps(
                {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "correct"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()


def test_financebench_judge_sends_credentials_only_to_judge_endpoint(judge_server):
    base_url, requests = judge_server

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="finance-judge-key",
            base_url=base_url,
        )
    )

    assert judgments == [("correct", "correct")]
    path, headers, body = requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer finance-judge-key"
    assert body["model"] == "judge-model"


def test_financebench_judge_recovers_from_three_transient_server_failures(judge_server):
    base_url, requests = judge_server

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "retry-model",
            api_key="finance-judge-key",
            base_url=base_url,
        )
    )

    assert judgments == [("correct", "correct")]
    assert len(requests) == 4


def test_financebench_judge_uses_dedicated_endpoint_and_key(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FakeAsyncOpenAI)

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )
    )

    assert judgments == [("correct", "correct")]
    assert _FakeAsyncOpenAI.constructor_kwargs["api_key"] == "judge-key"
    assert _FakeAsyncOpenAI.constructor_kwargs["base_url"] == "https://judge.example/v1"
    assert _FakeAsyncOpenAI.request_kwargs["model"] == "judge-model"


def test_financebench_judge_api_failure_is_returned_for_its_trial(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FailingAsyncOpenAI)

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )
    )

    assert len(judgments) == 1
    assert isinstance(judgments[0], RuntimeError)


def test_financebench_judge_allows_reasoning_before_label(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _ReasoningJudgeAsyncOpenAI)

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "reasoning-judge",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )
    )

    assert judgments == [("correct", "correct")]


def test_financebench_malformed_judge_response_is_returned_for_its_trial(monkeypatch):
    class MalformedAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="maybe"))])

    monkeypatch.setattr(finance_judge, "AsyncOpenAI", MalformedAsyncOpenAI)

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )
    )

    assert len(judgments) == 1
    assert isinstance(judgments[0], ValueError)


@pytest.mark.parametrize("response", ["The answer is correct.", "incorrect because it conflicts", ""])
def test_financebench_rejects_non_label_responses(monkeypatch, response):
    class NonLabelAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    monkeypatch.setattr(finance_judge, "AsyncOpenAI", NonLabelAsyncOpenAI)

    judgments = asyncio.run(
        finance_judge.judge_all(
            [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
            "judge-model",
            api_key="judge-key",
            base_url="https://judge.example/v1",
        )
    )

    assert len(judgments) == 1
    assert isinstance(judgments[0], ValueError)


def test_financebench_judge_failure_is_saved_per_trial_without_losing_other_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _MixedJudgeAsyncOpenAI)
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
    assert samples[1]["judge_error"]["exception_type"] == "TimeoutError"
    assert "judge_error" not in samples[1]["doc"]
    assert result["task_outcomes"]["FinanceBench"]["failure_counts"] == {"grader_infrastructure": 1}
    validate_result_document(result)


def test_financebench_all_judge_failures_return_no_accuracy(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FailingAsyncOpenAI)
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

    expected_cap = resolve_task_max_tokens(
        "FinanceBench",
        context_length=32768,
        requested_max_tokens=requested_max_tokens,
        prompt_lengths=manager.prompt_lengths,
    )
    assert model.generated_ids == [0]
    assert len(generated["examples"]) == 1
    assert model.request_kwargs[0]["max_new_tokens"] == expected_cap
    if requested_max_tokens is None:
        assert expected_cap > 4096
