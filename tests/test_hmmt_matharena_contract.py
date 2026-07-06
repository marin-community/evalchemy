import importlib.util
from types import SimpleNamespace

import pytest

MISSING_DEPS = [
    dep
    for dep in ("antlr4", "datasets", "lm_eval", "loguru", "regex", "sympy", "torch")
    if importlib.util.find_spec(dep) is None
]

pytestmark = pytest.mark.skipif(MISSING_DEPS, reason=f"missing benchmark dependencies: {', '.join(MISSING_DEPS)}")

HMMT_TASK_CONTRACTS = {
    "HMMT": ("MathArena/hmmt_feb_2025", 30),
    "HMMTFeb2023": ("MathArena/hmmt_feb_2023", 30),
    "HMMTFeb2024": ("MathArena/hmmt_feb_2024", 30),
    "HMMTNov2025": ("MathArena/hmmt_nov_2025", 30),
    "HMMTFeb2026": ("MathArena/hmmt_feb_2026", 33),
}


@pytest.fixture
def hmmt_modules():
    from eval.chat_benchmarks import hmmt_common
    from eval.chat_benchmarks.HMMT.eval_instruct import HMMTBenchmark
    from eval.chat_benchmarks.HMMTFeb2023.eval_instruct import HMMTFeb2023Benchmark
    from eval.chat_benchmarks.HMMTFeb2024.eval_instruct import HMMTFeb2024Benchmark
    from eval.chat_benchmarks.HMMTFeb2026.eval_instruct import HMMTFeb2026Benchmark
    from eval.chat_benchmarks.HMMTNov2025.eval_instruct import HMMTNov2025Benchmark
    from eval.task import TaskManager

    return SimpleNamespace(
        hmmt_common=hmmt_common,
        HMMTBenchmark=HMMTBenchmark,
        HMMTFeb2023Benchmark=HMMTFeb2023Benchmark,
        HMMTFeb2024Benchmark=HMMTFeb2024Benchmark,
        HMMTFeb2026Benchmark=HMMTFeb2026Benchmark,
        HMMTNov2025Benchmark=HMMTNov2025Benchmark,
        TaskManager=TaskManager,
    )


def test_hmmt_family_task_discovery_and_contracts(hmmt_modules):
    tasks = list(HMMT_TASK_CONTRACTS)
    manager = hmmt_modules.TaskManager(task_list=tasks)
    assert set(manager.available_tasks) == set(tasks)

    for task_name, (dataset_name, expected_rows) in HMMT_TASK_CONTRACTS.items():
        benchmark = manager.get_benchmark(task_name)
        assert benchmark is not None
        assert benchmark.__class__.__name__.replace("Benchmark", "") == task_name
        assert benchmark.dataset_name == dataset_name
        assert benchmark.EXPECTED_NUM_ROWS == expected_rows


def test_hmmt_load_questions_normalizes_and_debug_slices(monkeypatch, hmmt_modules):
    rows = [
        {"problem_idx": 1, "problem": "Problem 1", "answer": 7, "problem_type": ["algebra"]},
        {"problem_idx": 2, "problem": "Problem 2", "answer": "-\\frac{1}{21}", "problem_type": ["geometry"]},
        {"problem_idx": 3, "problem": "Problem 3", "answer": "0,i\\sqrt{6}", "problem_type": ["algebra"]},
    ]

    def fake_load_dataset(dataset_name, **kwargs):
        assert dataset_name == "MathArena/hmmt_feb_2026"
        assert kwargs == {"split": "train"}
        return rows

    monkeypatch.setattr(hmmt_modules.hmmt_common, "load_dataset", fake_load_dataset)
    benchmark = hmmt_modules.HMMTFeb2026Benchmark(debug=True)

    questions = benchmark.load_questions()

    assert len(questions) == 2
    assert questions[0]["id"] == "1"
    assert questions[0]["answer"] == "7"
    assert questions[0]["problem_type"] == ["algebra"]
    assert questions[1]["id"] == "2"
    assert questions[1]["answer"] == "-\\frac{1}{21}"


def test_hmmt_load_questions_checks_full_row_count(monkeypatch, hmmt_modules):
    rows = [{"problem_idx": 1, "problem": "Problem 1", "answer": 7}]

    monkeypatch.setattr(hmmt_modules.hmmt_common, "load_dataset", lambda *args, **kwargs: rows)

    with pytest.raises(ValueError, match="expected 33 rows, got 1"):
        hmmt_modules.HMMTFeb2026Benchmark().load_questions()


def test_hmmt_load_questions_passes_revision(monkeypatch, hmmt_modules):
    calls = []

    def fake_load_dataset(dataset_name, **kwargs):
        calls.append((dataset_name, kwargs))
        return [{"problem_idx": 1, "problem": "Problem 1", "answer": 7}]

    monkeypatch.setattr(hmmt_modules.hmmt_common, "load_dataset", fake_load_dataset)
    benchmark = hmmt_modules.HMMTFeb2026Benchmark(dataset_revision="abc123", debug=True)

    benchmark.load_questions()

    assert calls == [("MathArena/hmmt_feb_2026", {"split": "train", "revision": "abc123"})]


def test_hmmt_missing_required_fields_raise_clear_errors(hmmt_modules):
    benchmark = hmmt_modules.HMMTFeb2023Benchmark(debug=True)

    with pytest.raises(KeyError, match="missing 'problem'"):
        benchmark._normalize_example({"answer": "1"}, 0)

    with pytest.raises(KeyError, match="missing 'answer'"):
        benchmark._normalize_example({"problem": "Problem"}, 0)


def test_hmmt_generate_responses_preserves_metadata_and_scores(monkeypatch, hmmt_modules):
    rows = [
        {"problem_idx": 1, "problem": "Problem 1", "answer": "7", "problem_type": ["algebra"]},
        {"problem_idx": 2, "problem": "Problem 2", "answer": "8", "problem_type": ["geometry"]},
    ]

    monkeypatch.setattr(hmmt_modules.hmmt_common, "load_dataset", lambda *args, **kwargs: rows)
    benchmark = hmmt_modules.HMMTFeb2026Benchmark(debug=True, max_tokens=123, seed=[10, 20, 30, 40])
    benchmark.n_repeat = 2
    captured_batches = []

    def fake_compute(model, instances):
        captured_batches.append(instances)
        repeat_idx = instances[0].repeat_idx
        assert all(instance.repeat_idx == repeat_idx for instance in instances)
        return ["The answer is \\boxed{7}.", "The answer is \\boxed{8}."]

    class FakeModel:
        rank = 0

        def apply_chat_template(self, messages):
            return messages

    monkeypatch.setattr(benchmark, "compute", fake_compute)

    generated = benchmark.generate_responses(FakeModel())

    assert len(captured_batches) == 2
    for repeat_idx, batch in enumerate(captured_batches):
        assert len(batch) == 2
        assert [instance.repeat_idx for instance in batch] == [repeat_idx, repeat_idx]
        assert batch[0].args[0][0]["content"].startswith("Problem: Problem 1")
        assert batch[0].args[1]["max_new_tokens"] == 123
        assert batch[0].args[1]["seed"] == [10 + repeat_idx, 20 + repeat_idx, 30 + repeat_idx, 40 + repeat_idx]
        assert batch[0].metadata == {
            "problem_id": "1",
            "expected_answer": "7",
            "reference_solution": "",
            "dataset_name": "MathArena/hmmt_feb_2026",
            "problem_type": ["algebra"],
        }

    assert generated["examples"][0]["model_outputs"] == [
        "The answer is \\boxed{7}.",
        "The answer is \\boxed{7}.",
    ]
    assert generated["examples"][1]["model_outputs"] == [
        "The answer is \\boxed{8}.",
        "The answer is \\boxed{8}.",
    ]
    assert len(generated["examples"][0]["model_answers"]) == 2

    evaluated = benchmark.evaluate_responses(generated)

    assert evaluated["num_total"] == 2
    assert evaluated["num_repeat"] == 2
    assert evaluated["solved_avg"] == 2
    assert evaluated["accuracy_avg"] == 1
    assert evaluated["examples"][0]["label"] == [True, True]
    assert evaluated["examples"][1]["label"] == [True, True]
