import pytest

pytest.importorskip("antlr4")
pytest.importorskip("datasets")
pytest.importorskip("lm_eval")
pytest.importorskip("sympy")

from eval.chat_benchmarks.OlympiadBench import eval_instruct as olympiad
from eval.chat_benchmarks.OlympiadBench.auto_scoring_judge import AutoScoringJudge
from eval.task import TaskManager


def _row(**overrides):
    row = {
        "id": "math-1",
        "question": "What is 40 + 2?",
        "solution": "Add the numbers.",
        "final_answer": ["42"],
        "context": "Answer as an integer.",
        "modality": "Text-only",
        "difficulty": "easy",
        "is_multiple_answer": False,
        "unit": "",
        "answer_type": "Numerical",
        "error": 1e-8,
        "question_type": "Open-ended",
        "subfield": "Algebra",
        "subject": "maths",
        "language": "English",
    }
    row.update(overrides)
    return row


class _DeterministicLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.rank = 0
        self.world_size = 1
        self.instances = []
        self.messages = []

    def apply_chat_template(self, messages):
        self.messages.append(messages)
        return messages

    def generate_until(self, instances):
        self.instances = list(instances)
        return [self.outputs[instance.idx] for instance in instances]


def test_official_scorer_handles_tolerance_and_unordered_answers():
    scorer = AutoScoringJudge()

    assert scorer.judge("1.000", "1.000001", precision=1e-3)
    assert scorer.judge("\\boxed{\\frac{1}{2}}", "0.5")
    assert scorer.judge("1,2", "2,1")
    assert not scorer.judge("1.000", "1.1", precision=1e-3)


def test_normalize_example_filters_non_v1_rows():
    bench = olympiad.OlympiadBenchBenchmark()

    normalized = bench._normalize_example(_row(), "OE_TO_maths_en_COMP")
    assert normalized["answer"] == "42"
    assert normalized["subject"] == "Math"
    assert normalized["language"] == "English"

    assert (
        bench._normalize_example(_row(modality="Multimodal"), "OE_MM_maths_en_COMP")
        is None
    )
    assert (
        bench._normalize_example(
            _row(question_type="Theorem proof"), "TP_TO_maths_en_COMP"
        )
        is None
    )


def test_prompt_uses_context_and_does_not_leak_solution_or_answer():
    bench = olympiad.OlympiadBenchBenchmark()
    example = bench._normalize_example(
        _row(
            context="Use only arithmetic.",
            solution="SECRET_SOLUTION",
            final_answer=["SECRET_ANSWER"],
        ),
        "OE_TO_maths_en_COMP",
    )

    prompt = bench._build_prompt(example)

    assert "Context: Use only arithmetic." in prompt
    assert "What is 40 + 2?" in prompt
    assert "\\boxed{}" in prompt
    assert "SECRET_SOLUTION" not in prompt
    assert "SECRET_ANSWER" not in prompt


def test_generate_and_evaluate_builds_instances_and_scores_outputs(monkeypatch):
    rows = [
        _row(id="math-1", final_answer=["42"]),
        _row(id="math-2", question="What is 1 + 1?", final_answer=["2"]),
    ]

    def fake_load_dataset(dataset_name, subset, split, cache_dir):
        assert dataset_name == "Hothan/OlympiadBench"
        assert subset == "OE_TO_maths_en_COMP"
        assert split == "train"
        return rows

    monkeypatch.setattr(olympiad, "load_dataset", fake_load_dataset)

    bench = olympiad.OlympiadBenchBenchmark()
    bench.subsets = ["OE_TO_maths_en_COMP"]
    model = _DeterministicLM(
        ["Here is the result: \\boxed{42}", "Incorrectly, \\boxed{5}"]
    )

    generated = bench.generate_responses(model)
    scored = bench.evaluate_responses(generated)

    assert generated["benchmark_scope"] == "open_ended_text_only"
    assert generated["examples"][0]["model_output"] == "Here is the result: \\boxed{42}"
    assert generated["examples"][0]["model_answer"] == "42"
    assert model.messages[0][0]["content"].startswith("Context: Answer as an integer.")
    assert model.instances[0].request_type == "generate_until"
    assert scored["num_total"] == 2
    assert scored["num_solved"] == 1
    assert scored["accuracy"] == 0.5
    assert scored["accuracy_subset_OE_TO_maths_en_COMP"] == 0.5
    assert scored["accuracy_subject_Math"] == 0.5
    assert scored["accuracy_language_English"] == 0.5
    assert model.instances[0].task_name == "OlympiadBench"
    assert model.instances[0].metadata["problem_id"] == "math-1"


def test_num_samples_gt_one_fails_loudly():
    with pytest.raises(ValueError, match="single-sample"):
        olympiad.OlympiadBenchBenchmark(num_samples=2)


def test_evaluation_treats_non_finite_dataset_error_as_default_precision():
    bench = olympiad.OlympiadBenchBenchmark()
    results = {
        "examples": [
            {
                "id": "math-precision",
                "answer": "1",
                "model_answer": "1.000000001",
                "error": float("nan"),
                "subset": "OE_TO_maths_en_COMP",
                "subject": "Math",
                "language": "English",
            }
        ]
    }

    scored = bench.evaluate_responses(results)

    assert scored["num_solved"] == 1
    assert scored["examples"][0]["is_correct"] is True


def test_extract_answer_handles_nested_boxed_fallback():
    bench = olympiad.OlympiadBenchBenchmark()

    assert bench.extract_answer("Final: \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert bench.extract_answer("No boxed answer") == ""


def test_task_manager_registers_olympiadbench_by_name():
    task_manager = TaskManager(task_list=["OlympiadBench"])
    benchmark = task_manager.get_benchmark("OlympiadBench")

    assert task_manager.available_tasks == ["OlympiadBench"]
    assert type(benchmark).__name__ == "OlympiadBenchBenchmark"
    assert benchmark.dataset_name == "Hothan/OlympiadBench"
    assert benchmark.subsets == olympiad.DEFAULT_OPEN_ENDED_TEXT_SUBSETS
