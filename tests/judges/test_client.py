import json
from types import SimpleNamespace

import pytest

from eval.judges import JudgeClient, JudgeConfig, JudgeRequest, ProofJudgeResult
from eval.judges.client import JudgeResponseError


def _proof_payload():
    return {
        "score": 1.0,
        "verdict": "correct",
        "reasoning": "The proof establishes the requested claim.",
        "issues": [],
        "confidence": 0.9,
        "logical_validity": 1.0,
        "completeness": 1.0,
        "rigor": 1.0,
        "final_claim_established": True,
        "fatal_gap": None,
        "hallucinated_theorem": False,
        "rubric_points": [{"name": "claim", "score": 1.0, "explanation": "satisfied"}],
    }


def _request():
    return JudgeRequest(
        task_name="IMO2025Proof",
        problem_id="p1",
        prompt="Prove that 1 + 1 = 2.",
        response="By Peano arithmetic, 1 + 1 = 2.",
        reference_answer="2",
    )


class _FakeResponses:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(
            id="resp_123",
            output_text=json.dumps(_proof_payload()),
            usage={"input_tokens": 10, "output_tokens": 20},
        )


class _FakeCompletions:
    def __init__(self, calls, content):
        self.calls = calls
        self.content = content

    def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(
            id="chatcmpl_123",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )


def test_openai_responses_path_uses_strict_json_schema(monkeypatch):
    calls = []
    client_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.responses = _FakeResponses(calls)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = JudgeConfig.from_model("gpt-5.5", reasoning_effort="high")
    result = JudgeClient(config, openai_client_factory=FakeOpenAI).grade(_request(), ProofJudgeResult)

    assert client_kwargs["api_key"] == "sk-test"
    assert "base_url" not in client_kwargs
    assert result.verdict == "correct"
    assert result.judge_model == "gpt-5.5"
    assert result.judge_provider == "openai"
    assert result.provider_response_id == "resp_123"
    assert result.usage == {"input_tokens": 10, "output_tokens": 20}

    payload = calls[0]
    assert payload["model"] == "gpt-5.5"
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["required"]


def test_deepseek_chat_path_uses_base_url_json_mode_and_thinking(monkeypatch):
    calls = []
    client_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=_FakeCompletions(calls, json.dumps(_proof_payload())))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    config = JudgeConfig.from_model("deepseek-v4-pro")
    result = JudgeClient(config, openai_client_factory=FakeOpenAI).grade(_request(), ProofJudgeResult)

    assert client_kwargs["api_key"] == "sk-test"
    assert client_kwargs["base_url"] == "https://api.deepseek.com"
    assert result.verdict == "correct"
    assert result.judge_model == "deepseek-v4-pro"
    assert result.judge_provider == "deepseek"
    assert result.provider_response_id == "chatcmpl_123"

    payload = calls[0]
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "high"
    assert payload["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "json" in payload["messages"][0]["content"].lower()


def test_invalid_json_raises_classified_error(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_FakeCompletions([], "not json"))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    config = JudgeConfig.from_model("deepseek-v4-pro")
    client = JudgeClient(config, openai_client_factory=FakeOpenAI)

    with pytest.raises(JudgeResponseError):
        client.grade(_request(), ProofJudgeResult)


def test_proof_schema_is_strict_for_nested_openai_objects():
    schema = ProofJudgeResult.json_schema()

    assert schema["additionalProperties"] is False
    rubric_items = schema["properties"]["rubric_points"]["items"]
    assert rubric_items["additionalProperties"] is False
    assert set(rubric_items["required"]) == {"name", "score", "explanation"}


def test_judge_request_renders_real_grading_context_and_json_instruction():
    request = JudgeRequest(
        task_name="IMO2025Proof",
        problem_id="p2",
        prompt="Prove that every even prime is 2.",
        response="Let p be an even prime. Since p is even, p = 2k. If k > 1 then p is composite.",
        reference_solution="The only even prime is 2 because any larger even integer is divisible by 2.",
        rubric={"logical_validity": "must justify no larger even prime exists"},
    )

    messages = request.to_messages("ProofJudgeResult")

    assert messages[0]["role"] == "system"
    assert "valid JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "every even prime is 2" in messages[1]["content"]
    assert "logical_validity" in messages[1]["content"]
    assert "Return JSON only" in messages[1]["content"]


def test_proof_result_normalizes_legacy_rubric_point_dict():
    payload = _proof_payload()
    payload["rubric_points"] = {"claim": 1.0, "rigor": "minor gap"}

    result = ProofJudgeResult.from_dict(payload)

    assert result.rubric_points == [
        {"name": "claim", "score": 1.0, "explanation": "1.0"},
        {"name": "rigor", "score": None, "explanation": "minor gap"},
    ]
