import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from eval.chat_benchmarks.FinanceBench import judge as finance_judge
from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
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


class _CandidateModel:
    rank = 0
    world_size = 1

    def __init__(self):
        self.generated_ids = []

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, instances):
        self.generated_ids.extend(instance.idx for instance in instances)
        return ["candidate answer" for _ in instances]


@pytest.fixture
def judge_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
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


def test_financebench_judge_api_failure_propagates(monkeypatch):
    monkeypatch.setattr(finance_judge, "AsyncOpenAI", _FailingAsyncOpenAI)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        asyncio.run(
            finance_judge.judge_all(
                [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
                "judge-model",
                api_key="judge-key",
                base_url="https://judge.example/v1",
            )
        )


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


def test_financebench_malformed_judge_response_propagates(monkeypatch):
    class MalformedAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="maybe"))])

    monkeypatch.setattr(finance_judge, "AsyncOpenAI", MalformedAsyncOpenAI)

    with pytest.raises(ValueError, match="unrecognized FinanceBench judgment"):
        asyncio.run(
            finance_judge.judge_all(
                [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
                "judge-model",
                api_key="judge-key",
                base_url="https://judge.example/v1",
            )
        )


@pytest.mark.parametrize("response", ["The answer is correct.", "incorrect because it conflicts", ""])
def test_financebench_rejects_non_label_responses(monkeypatch, response):
    class NonLabelAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    monkeypatch.setattr(finance_judge, "AsyncOpenAI", NonLabelAsyncOpenAI)

    with pytest.raises(ValueError, match="unrecognized FinanceBench judgment"):
        asyncio.run(
            finance_judge.judge_all(
                [{"question": "Revenue?", "answer": "$10", "model_output": "$10"}],
                "judge-model",
                api_key="judge-key",
                base_url="https://judge.example/v1",
            )
        )


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
