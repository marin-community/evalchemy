import asyncio
from types import SimpleNamespace

import pytest

from eval.graders import answer_equivalence
from eval.graders.answer_equivalence import EquivalenceJudgment, JudgeConfig, JudgeLabel
from eval.graders.simpleqa import SimpleQARequest, judge_simpleqa


class _SimpleQAJudge:
    response = "A"
    request_kwargs = None

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def create(self, **kwargs):
        type(self).request_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=type(self).response))])


@pytest.mark.parametrize(
    ("response", "expected_label"),
    [("A", JudgeLabel.CORRECT), ("b", JudgeLabel.INCORRECT), (" C\n", JudgeLabel.NOT_ATTEMPTED)],
)
def test_simpleqa_judge_maps_canonical_labels(monkeypatch, response, expected_label):
    _SimpleQAJudge.response = response
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _SimpleQAJudge)
    config = JudgeConfig("judge-model", "https://judge.example/v1", "judge-key")

    judgments = asyncio.run(
        judge_simpleqa(
            [SimpleQARequest("Which city?", "Paris", "The answer is Paris.")],
            config,
        )
    )

    assert judgments == [EquivalenceJudgment(expected_label, response.strip())]
    prompt = _SimpleQAJudge.request_kwargs["messages"][0]["content"]
    assert "Question: Which city?" in prompt
    assert "Gold target: Paris" in prompt
    assert "Predicted answer: The answer is Paris." in prompt


@pytest.mark.parametrize("response", ["A because it is correct", "CORRECT", "", "D"])
def test_simpleqa_judge_rejects_noncanonical_responses(monkeypatch, response):
    _SimpleQAJudge.response = response
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _SimpleQAJudge)
    config = JudgeConfig("judge-model", "https://judge.example/v1", "judge-key")

    judgments = asyncio.run(judge_simpleqa([SimpleQARequest("Question", "target", "answer")], config))

    assert len(judgments) == 1
    assert isinstance(judgments[0], ValueError)
