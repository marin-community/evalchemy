import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from eval.graders import answer_equivalence


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


def test_math_answers_equivalent_accepts_symbolically_equal_latex():
    assert answer_equivalence.math_answers_equivalent(r"\frac{1}{2}", ["0.5"])


def test_judge_config_resolves_normalized_environment_without_exposing_key(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")

    config = answer_equivalence.JudgeConfig.resolve(judge_model="auto")

    assert config.model == "judge-model"
    assert config.base_url == "https://judge.example/v1"
    assert config.api_key == "secret-key"
    assert "secret-key" not in repr(config)


def test_judge_config_does_not_fall_back_to_candidate_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "candidate-key")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="JUDGE_API_KEY"):
        answer_equivalence.JudgeConfig.resolve()


@pytest.mark.parametrize(
    "base_url",
    [
        "judge.example/v1",
        "https://secret@judge.example/v1",
        "https://judge.example/v1?token=secret",
        "https://judge.example/v1#secret",
    ],
)
def test_judge_config_rejects_unsafe_endpoint_urls(base_url):
    with pytest.raises(ValueError, match="JUDGE_BASE_URL"):
        answer_equivalence.JudgeConfig("judge-model", base_url, "judge-key")


def test_judge_equivalence_uses_dedicated_endpoint_and_credentials(monkeypatch):
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _FakeAsyncOpenAI)
    config = answer_equivalence.JudgeConfig(
        model="judge-model",
        base_url="https://judge.example/v1",
        api_key="judge-key",
    )
    requests = [
        answer_equivalence.EquivalenceRequest(
            question="How many centimeters are in one meter?",
            reference_answers=("100 cm",),
            candidate_answer="1 m",
        )
    ]

    judgments = asyncio.run(answer_equivalence.judge_equivalence(requests, config))

    assert judgments == [
        answer_equivalence.EquivalenceJudgment(
            label=answer_equivalence.JudgeLabel.CORRECT,
            raw="correct",
        )
    ]
    assert _FakeAsyncOpenAI.constructor_kwargs["api_key"] == "judge-key"
    assert _FakeAsyncOpenAI.constructor_kwargs["base_url"] == "https://judge.example/v1"
    assert _FakeAsyncOpenAI.request_kwargs["model"] == "judge-model"
    assert "100 cm" in _FakeAsyncOpenAI.request_kwargs["messages"][0]["content"]


def test_judge_equivalence_retries_transient_server_failures(judge_server):
    base_url, requests = judge_server
    request = answer_equivalence.EquivalenceRequest("Question", ("reference",), "candidate")
    config = answer_equivalence.JudgeConfig("retry-model", base_url, "judge-key")

    judgments = asyncio.run(answer_equivalence.judge_equivalence([request], config))

    assert judgments == [
        answer_equivalence.EquivalenceJudgment(answer_equivalence.JudgeLabel.CORRECT, "correct")
    ]
    assert len(requests) == 4
    path, headers, body = requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer judge-key"
    assert body["model"] == "retry-model"


def test_judge_equivalence_retries_empty_reasoning_completion(monkeypatch):
    class ReasoningJudgeAsyncOpenAI(_FakeAsyncOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.requests = 0

        async def create(self, **kwargs):
            self.requests += 1
            content = "correct" if self.requests == 3 else ""
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", ReasoningJudgeAsyncOpenAI)
    request = answer_equivalence.EquivalenceRequest("Question", ("reference",), "candidate")
    config = answer_equivalence.JudgeConfig("reasoning-judge", "https://judge.example/v1", "judge-key")

    judgments = asyncio.run(answer_equivalence.judge_equivalence([request], config))

    assert judgments == [
        answer_equivalence.EquivalenceJudgment(answer_equivalence.JudgeLabel.CORRECT, "correct")
    ]


@pytest.mark.parametrize("response", ["maybe", "The answer is correct.", "incorrect because it conflicts", ""])
def test_judge_equivalence_rejects_non_label_responses(monkeypatch, response):
    class NonLabelAsyncOpenAI(_FakeAsyncOpenAI):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", NonLabelAsyncOpenAI)
    request = answer_equivalence.EquivalenceRequest("Question", ("reference",), "candidate")
    config = answer_equivalence.JudgeConfig("judge-model", "https://judge.example/v1", "judge-key")

    judgments = asyncio.run(answer_equivalence.judge_equivalence([request], config))

    assert len(judgments) == 1
    assert isinstance(judgments[0], ValueError)
