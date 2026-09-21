import pytest

from eval.chat_benchmarks.SimpleQA.eval_instruct import DATASET_SIZE, SimpleQABenchmark
from eval.chat_benchmarks.SimpleQAMini.eval_instruct import MINI_DATASET_SIZE, SimpleQAMiniBenchmark
from eval.graders.answer_equivalence import EquivalenceJudgment, JudgeLabel


class _CandidateModel:
    rank = 0
    world_size = 1

    def __init__(self):
        self.instances = []

    @staticmethod
    def apply_chat_template(messages):
        return messages

    def generate_until(self, instances):
        self.instances.extend(instances)
        return ["candidate answer" for _ in instances]


def test_simpleqa_loads_the_canonical_full_dataset():
    benchmark = SimpleQABenchmark(judge_api_key="judge-key")

    questions = benchmark.load_questions()

    assert len(questions) == DATASET_SIZE
    assert questions[0] == {
        "source_index": 0,
        "question": "Who received the IEEE Frank Rosenblatt Award in 2010?",
        "answer": "Michio Sugeno",
        "metadata": {
            "topic": "Science and technology",
            "answer_type": "Person",
            "urls": [
                "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
                "https://ieeexplore.ieee.org/author/37271220500",
                "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
                "https://www.nxtbook.com/nxtbooks/ieee/awards_2010/index.php?startid=21#/p/20",
            ],
        },
    }


def test_simpleqa_mini_uses_the_reference_seeded_subset():
    benchmark = SimpleQAMiniBenchmark(judge_api_key="judge-key")

    questions = benchmark.load_questions()

    assert len(questions) == MINI_DATASET_SIZE
    assert [question["source_index"] for question in questions[:10]] == [
        3155,
        3445,
        331,
        2121,
        4188,
        3980,
        3317,
        2484,
        3904,
        2933,
    ]
    assert len({question["source_index"] for question in questions}) == MINI_DATASET_SIZE


@pytest.mark.parametrize("benchmark_class", [SimpleQABenchmark, SimpleQAMiniBenchmark])
def test_simpleqa_generation_obeys_the_shared_sample_limit(benchmark_class):
    benchmark = benchmark_class(judge_api_key="judge-key")
    benchmark.set_evaluation_limits(limit=1)
    model = _CandidateModel()

    generated = benchmark.generate_responses(model)

    assert len(generated["examples"]) == 1
    assert model.instances[0].idx == generated["examples"][0]["source_index"]
    assert model.instances[0].args[0] == [{"role": "user", "content": generated["examples"][0]["question"]}]


def test_simpleqa_reports_canonical_aggregate_metrics(monkeypatch):
    async def fake_judge(requests, _config):
        labels = [JudgeLabel.CORRECT, JudgeLabel.INCORRECT, JudgeLabel.NOT_ATTEMPTED]
        return [EquivalenceJudgment(label, letter) for label, letter in zip(labels, "ABC", strict=True)]

    benchmark = SimpleQABenchmark(judge_api_key="judge-key")
    monkeypatch.setitem(benchmark.evaluate_responses.__globals__, "judge_simpleqa", fake_judge)
    examples = [
        {"question": f"Question {index}", "answer": "answer", "model_output": "candidate"} for index in range(3)
    ]

    result = benchmark.evaluate_responses({"examples": examples, "judge_model": "judge-model"})

    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["accuracy_given_attempted"] == 0.5
    assert result["f1"] == pytest.approx(0.4)
    assert result["num_correct"] == 1
    assert result["num_incorrect"] == 1
    assert result["num_not_attempted"] == 1
    assert result["judge_coverage"] == 1.0
    assert [example["judge_label"] for example in examples] == ["correct", "incorrect", "not_attempted"]
