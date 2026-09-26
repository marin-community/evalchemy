# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry-wide coverage for the uniform per-sample metrics contract.

Every registered custom benchmark appears exactly once below: either with a
dummy grading case that drives its real grader, or in ``NO_SAMPLE_RECORDS``
because it persists no sample records at all. Adding a benchmark without
deciding which side it belongs on fails the coverage test.
"""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval.contracts.grading import GenerationArtifactManifest
from eval.contracts.sample_results import (
    SAMPLE_METRICS_FIELD,
    SampleMetricsError,
    record_sample_metrics,
    sample_metric_fields,
    validate_sample_metrics,
)
from eval.eval import CHAT_BENCHMARK_ROUTE, evaluate
from eval.graders.answer_equivalence import EquivalenceJudgment, JudgeLabel
from eval.sample_logging import canonicalize_samples
from eval.task import BaseBenchmark, TaskManager

CUSTOM_BENCHMARK_ROOT = Path("eval/chat_benchmarks")

NO_SAMPLE_RECORDS = frozenset(
    {
        # These benchmarks return no examples for the sample writer. The driver
        # retains their aggregate score and logs the missing sample records.
        "BigCodeBench",
        "HumanEval",
        "LiveBench",
        "MTBench",
        "MixEval",
        "MultiPLE",
        "RepoBench",
        "SWEbench",
        "WildBench",
        "alpaca_eval",
        "zeroeval",
    }
)

PreparedCase = tuple[BaseBenchmark, dict[str, Any]]
BuildExamples = Callable[[BaseBenchmark], Sequence[Mapping[str, Any]]]
PatchBoundary = Callable[[pytest.MonkeyPatch, BaseBenchmark], None]


@dataclass(frozen=True)
class GradingCase:
    """A benchmark's dummy grading input and the per-sample metrics it owes."""

    prepare: Callable[[pytest.MonkeyPatch, Path], PreparedCase]
    metrics: tuple[str, ...]


def _load_benchmark(task_name: str, **kwargs: Any) -> BaseBenchmark:
    """Load one registered benchmark the way the driver does."""
    manager = TaskManager(task_list=[task_name], **kwargs)
    failure = manager.load_failures.get(task_name)
    if isinstance(failure, ModuleNotFoundError):
        pytest.skip(f"{task_name} needs its optional extra: {failure}")
    assert failure is None, failure
    return manager.get_benchmark(task_name)


def _examples_case(
    task_name: str,
    build_examples: BuildExamples,
    *,
    patch: PatchBoundary | None = None,
) -> Callable[[pytest.MonkeyPatch, Path], PreparedCase]:
    """Build a case whose generation result is the shared ``{"examples": [...]}`` shape."""

    def prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
        del tmp_path  # This shape needs no files on disk.
        benchmark = _load_benchmark(task_name)
        if patch is not None:
            patch(monkeypatch, benchmark)
        return benchmark, {"examples": [dict(example) for example in build_examples(benchmark)]}

    return prepare


def _repeated(answer: str, benchmark: BaseBenchmark) -> list[str]:
    """Return one model answer per repetition the benchmark grades."""
    return [answer] * benchmark.n_repeat


def _patch_single_example_grader(monkeypatch: pytest.MonkeyPatch, benchmark: BaseBenchmark) -> None:
    """Stand in for the per-problem code-execution sandbox the grader shells out to."""
    monkeypatch.setattr(
        benchmark,
        "evaluate_single_example",
        lambda example: {"content": example["model_answer"], "correctness": True, "reason": "stub"},
    )


def _passing_sandbox(task_id: str, _sample, _language, _timeout, _tmp_dir, completion_id: int) -> dict[str, Any]:
    """Report a passing run without launching the real code sandbox."""
    return {"task_id": task_id, "completion_id": completion_id, "passed": True}


def _patch_code_sandbox(monkeypatch: pytest.MonkeyPatch, benchmark: BaseBenchmark) -> None:
    """Replace the sandbox launch so the grader's own aggregation still runs."""
    evaluation = benchmark.evaluate_responses.__globals__["evaluate_functional_correctness"].__globals__
    monkeypatch.setitem(evaluation, "check_correctness", _passing_sandbox)


def _patch_judge(monkeypatch: pytest.MonkeyPatch, benchmark: BaseBenchmark) -> None:
    """Stand in for the FinanceBench judge's HTTP calls."""

    async def judge_equivalence(requests, *_args, **_kwargs):
        return [EquivalenceJudgment(JudgeLabel.CORRECT, "graded") for _ in requests]

    monkeypatch.setitem(benchmark.evaluate_responses.__globals__, "judge_equivalence", judge_equivalence)


def _patch_simpleqa_judge(monkeypatch: pytest.MonkeyPatch, benchmark: BaseBenchmark) -> None:
    """Stand in for the SimpleQA classifier's HTTP calls."""

    async def judge_simpleqa(requests, *_args, **_kwargs):
        return [EquivalenceJudgment(JudgeLabel.CORRECT, "A") for _ in requests]

    monkeypatch.setitem(benchmark.evaluate_responses.__globals__, "judge_simpleqa", judge_simpleqa)


def _financebench_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Grade one FinanceBench answer with the judge boundary stubbed out."""
    del tmp_path
    monkeypatch.setenv("JUDGE_API_KEY", "conformance-test")
    benchmark = _load_benchmark("FinanceBench")
    _patch_judge(monkeypatch, benchmark)
    return benchmark, {"examples": [{"question": "q", "answer": "a", "model_answer": "a"}]}


def _humanevalplus_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Grade one HumanEvalPlus completion through the real functional-correctness pass."""
    problem = {"task_id": "python/0", "prompt": "def answer():\n", "test": "assert answer() == 42"}
    (tmp_path / "humanevalplus-python.jsonl").write_text(json.dumps(problem) + "\n")
    benchmark = _load_benchmark("HumanEvalPlus", data_dir=str(tmp_path), num_workers=1)
    _patch_code_sandbox(monkeypatch, benchmark)
    example = {**problem, "language": "python", "generation": "def answer():\n    return 42"}
    artifacts = GenerationArtifactManifest.temporary()
    artifacts.write_jsonl("generated-python", "generated_python.jsonl", [example], expected_count=1)
    return benchmark, {"examples": [example], "artifacts": artifacts}


def _mbppplus_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Grade one MBPPPlus completion through the real functional-correctness pass."""
    problem = {"task_id": "python/0", "prompt": "def answer():\n", "test": "assert answer() == 42"}
    (tmp_path / "mbppplus.jsonl").write_text(json.dumps(problem) + "\n")
    benchmark = _load_benchmark("MBPPPlus", data_dir=str(tmp_path), num_workers=1)
    _patch_code_sandbox(monkeypatch, benchmark)
    example = {**problem, "generation": "def answer():\n    return 42"}
    artifacts = GenerationArtifactManifest.temporary()
    artifacts.write_jsonl("generated-python", "generated_python.jsonl", [example], expected_count=1)
    return benchmark, {"examples": [example], "artifacts": artifacts}


def _mbpp_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Grade one MBPP completion through the real functional-correctness pass."""
    problem = {"task_id": "python/0", "prompt": "def answer():\n", "test": ["assert answer() == 42"]}
    (tmp_path / "mbpp_test.jsonl").write_text(json.dumps(problem) + "\n")
    benchmark = _load_benchmark("MBPP", data_dir=str(tmp_path))
    _patch_code_sandbox(monkeypatch, benchmark)
    example = {**problem, "generation": "def answer():\n    return 42"}
    temp_dir_obj = tempfile.TemporaryDirectory()
    (Path(temp_dir_obj.name) / "mbpp.jsonl").write_text(json.dumps(example) + "\n")
    return benchmark, {
        "temp_dir_obj": temp_dir_obj,
        "examples": [example],
        "num_examples": 1,
        "total_examples": 1,
    }


def _cruxeval_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Generate and grade one CruxEval row in both directions."""
    del monkeypatch
    row = {"id": "sample_0", "code": "def f(x): return x + 1", "input": "1", "output": "2"}
    (tmp_path / "cruxeval.jsonl").write_text(json.dumps(row) + "\n")
    benchmark = _load_benchmark("CruxEval", data_dir=str(tmp_path))
    return benchmark, benchmark.generate_responses(_AnsweringModel())


def _ifeval_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    """Grade one IFEval response against a real verifiable instruction."""
    del monkeypatch, tmp_path
    benchmark = _load_benchmark("IFEval")
    example = {
        "key": 0,
        "prompt": "Reply without commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
        "response": "No commas here",
    }
    temp_dir_obj = tempfile.TemporaryDirectory()
    (Path(temp_dir_obj.name) / "ifeval.jsonl").write_text(json.dumps(example) + "\n")
    return benchmark, {
        "temp_dir_obj": temp_dir_obj,
        "examples": [example],
        "num_examples": 1,
        "total_examples": 1,
    }


def _ifbench_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PreparedCase:
    del tmp_path
    benchmark = _load_benchmark("IFBench")
    passing = {
        "key": 0,
        "prompt": "Reply without whitespace.",
        "instruction_id_list": ["format:no_whitespace"],
        "kwargs": [{}],
    }
    failing = {
        "key": 1,
        "prompt": "Reply with one word and no whitespace.",
        "instruction_id_list": ["format:no_whitespace"],
        "kwargs": [{}],
    }
    monkeypatch.setattr(benchmark, "load_questions", lambda: [passing, failing])
    return benchmark, benchmark.generate_responses(_IFBenchModel())


class _AnsweringModel:
    """Minimal ``LM`` stand-in that answers every request with the gold value."""

    rank = 0
    world_size = 1

    @staticmethod
    def apply_chat_template(messages):
        return messages

    @staticmethod
    def generate_until(instances):
        return ["[ANSWER] 2 [/ANSWER]" for _ in instances]


class _IFBenchModel:
    rank = 0
    world_size = 1

    @staticmethod
    def apply_chat_template(messages):
        return messages

    @staticmethod
    def generate_until(instances):
        assert len(instances) == 2
        return ["NoWhitespace", "two words"]


GRADING_CASES: dict[str, GradingCase] = {
    "AIME24": GradingCase(
        _examples_case(
            "AIME24",
            lambda benchmark: [{"problem": "p", "expected_answer": "4", "model_answers": _repeated("4", benchmark)}],
        ),
        ("accuracy",),
    ),
    "AIME25": GradingCase(
        _examples_case(
            "AIME25",
            lambda benchmark: [{"problem": "p", "answer": "4", "model_answers": _repeated("4", benchmark)}],
        ),
        ("accuracy",),
    ),
    "AIW": GradingCase(
        _examples_case("AIW", lambda _benchmark: [{"id": 577, "right_answer": "4", "model_answer": "4"}]),
        ("accuracy",),
    ),
    "AMC23": GradingCase(
        _examples_case(
            "AMC23",
            lambda benchmark: [{"problem": "p", "answer": "4", "model_answers": _repeated("4", benchmark)}],
        ),
        ("accuracy",),
    ),
    "CodeElo": GradingCase(
        _examples_case(
            "CodeElo",
            lambda _benchmark: [{"model_outputs": ["c"], "model_answers": ["c"], "rating": 800, "difficulty": "easy"}],
            patch=_patch_single_example_grader,
        ),
        ("accuracy",),
    ),
    "CodeForces": GradingCase(
        _examples_case(
            "CodeForces",
            lambda _benchmark: [{"model_outputs": ["c"], "model_answers": ["c"], "rating": 800, "difficulty": "easy"}],
            patch=_patch_single_example_grader,
        ),
        ("accuracy",),
    ),
    "CruxEval": GradingCase(_cruxeval_case, ("pass_rate",)),
    "FinanceBench": GradingCase(_financebench_case, ("accuracy", "not_attempted", "judge_failed")),
    "GPQADiamond": GradingCase(
        _examples_case(
            "GPQADiamond",
            lambda benchmark: [{"Question": "q", "answer": "A", "model_answers": _repeated("A", benchmark)}],
        ),
        ("accuracy",),
    ),
    "GSM8KPerturbed": GradingCase(
        _examples_case(
            "GSM8KPerturbed",
            lambda _benchmark: [{"id": "1", "task": "gsm8k-clean", "output": "#### 4", "answer": "4"}],
        ),
        ("accuracy", "no_answer"),
    ),
    "HLE": GradingCase(
        _examples_case(
            "HLE",
            lambda benchmark: [{"question": "q", "answer": "A", "model_answers": _repeated("A", benchmark)}],
        ),
        ("accuracy",),
    ),
    "HMMT": GradingCase(
        _examples_case(
            "HMMT",
            lambda benchmark: [
                {"problem": "p", "answer": "4", "model_answers": _repeated("4", benchmark), "label": []}
            ],
        ),
        ("accuracy",),
    ),
    "HumanEvalPlus": GradingCase(_humanevalplus_case, ("pass_rate",)),
    "IFEval": GradingCase(_ifeval_case, ("prompt_level_strict", "prompt_level_loose")),
    "IFBench": GradingCase(
        _ifbench_case,
        (
            "strict_prompt_accuracy",
            "loose_prompt_accuracy",
            "strict_instruction_accuracy",
            "loose_instruction_accuracy",
        ),
    ),
    "JEEBench": GradingCase(
        _examples_case(
            "JEEBench",
            lambda benchmark: [
                {"question": "q", "gold": "A", "model_answers": _repeated("A", benchmark), "type": "MCQ"}
            ],
        ),
        ("accuracy",),
    ),
    "LiveCodeBench": GradingCase(
        _examples_case(
            "LiveCodeBench",
            lambda _benchmark: [{"model_outputs": ["c"], "model_answers": ["c"], "difficulty": "easy"}],
            patch=_patch_single_example_grader,
        ),
        ("accuracy",),
    ),
    "LiveCodeBenchv5": GradingCase(
        _examples_case(
            "LiveCodeBenchv5",
            lambda _benchmark: [{"model_outputs": ["c"], "model_answers": ["c"], "difficulty": "easy"}],
            patch=_patch_single_example_grader,
        ),
        ("accuracy",),
    ),
    "LiveCodeBenchv5_official": GradingCase(
        _examples_case(
            "LiveCodeBenchv5_official",
            lambda _benchmark: [{"model_outputs": ["c"], "model_answers": ["c"], "difficulty": "easy"}],
            patch=_patch_single_example_grader,
        ),
        ("accuracy",),
    ),
    "MATH500": GradingCase(
        _examples_case("MATH500", lambda _benchmark: [{"problem": "p", "answer": "4", "model_answer": "4"}]),
        ("accuracy",),
    ),
    "MBPP": GradingCase(_mbpp_case, ("pass_rate",)),
    "MBPPPlus": GradingCase(_mbppplus_case, ("pass_rate",)),
    "MMLUPro": GradingCase(
        _examples_case(
            "MMLUPro",
            lambda _benchmark: [{"question": "q", "category": "math", "model_answer": "A", "answer": "A"}],
        ),
        ("accuracy",),
    ),
    "MRCR": GradingCase(
        _examples_case(
            "MRCR",
            lambda _benchmark: [{"score": 1.0, "prefix_hit": 1.0, "mrcr_bin_upper": 8192, "n_needles": 2}],
        ),
        ("accuracy", "prefix_hit"),
    ),
    "NUPA": GradingCase(
        _examples_case(
            "NUPA",
            lambda _benchmark: [
                {
                    "task_name": "max_Float_Float_Float",
                    "length_bucket": "S",
                    "answer_format": "Float",
                    "answer": "9.9",
                    "output": "9.9",
                }
            ],
        ),
        ("exact_match", "digit_match", "dlength", "format_valid_rate", "no_answer_rate"),
    ),
    "NUPA5K": GradingCase(
        _examples_case(
            "NUPA5K",
            lambda _benchmark: [
                {
                    "task_name": "max_Float_Float_Float",
                    "length_bucket": "S",
                    "answer_format": "Float",
                    "answer": "9.9",
                    "output": "9.9",
                }
            ],
        ),
        ("exact_match", "digit_match", "dlength", "format_valid_rate", "no_answer_rate"),
    ),
    # OlympiadBench repeats each problem while OlympiadBenchFull pins one
    # repetition, so the pair covers both of the shared grader's paths.
    "OlympiadBench": GradingCase(
        _examples_case(
            "OlympiadBench",
            lambda benchmark: [{"problem": "p", "answer": ["17"], "model_answers": _repeated("17", benchmark)}],
        ),
        ("accuracy",),
    ),
    "OlympiadBenchDeterministic": GradingCase(
        _examples_case(
            "OlympiadBenchDeterministic",
            lambda _benchmark: [{"problem": "p", "answer": ["17"], "model_answer": "17"}],
        ),
        ("accuracy",),
    ),
    "OlympiadBenchFull": GradingCase(
        _examples_case(
            "OlympiadBenchFull",
            lambda _benchmark: [{"problem": "p", "answer": ["17"], "model_answer": "17"}],
        ),
        ("accuracy",),
    ),
    "SimpleQA": GradingCase(
        _examples_case(
            "SimpleQA",
            lambda _benchmark: [{"question": "q", "answer": "a", "model_output": "a"}],
            patch=_patch_simpleqa_judge,
        ),
        ("accuracy", "incorrect", "not_attempted", "judge_failed"),
    ),
    "SimpleQAMini": GradingCase(
        _examples_case(
            "SimpleQAMini",
            lambda _benchmark: [{"question": "q", "answer": "a", "model_output": "a"}],
            patch=_patch_simpleqa_judge,
        ),
        ("accuracy", "incorrect", "not_attempted", "judge_failed"),
    ),
}


def test_every_registered_benchmark_is_covered_by_the_per_sample_metrics_contract():
    registered = {path.parent.name for path in CUSTOM_BENCHMARK_ROOT.glob("*/eval_instruct.py")}

    assert registered == set(GRADING_CASES) | NO_SAMPLE_RECORDS
    assert not set(GRADING_CASES) & NO_SAMPLE_RECORDS


@pytest.mark.parametrize("task_name", sorted(GRADING_CASES))
def test_graded_benchmark_samples_persist_their_per_sample_metrics(task_name, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "conformance-test")
    monkeypatch.setenv("JUDGE_API_KEY", "conformance-test")
    case = GRADING_CASES[task_name]
    benchmark, generation_result = case.prepare(monkeypatch, tmp_path)

    scored_result = benchmark.evaluate_responses(generation_result)
    records = canonicalize_samples(task_name, benchmark.to_samples(generation_result, scored_result))

    assert records
    for record in records:
        assert set(record[SAMPLE_METRICS_FIELD]) == set(case.metrics)
        assert all(isinstance(record[name], float) for name in case.metrics)


def test_ifbench_sample_contains_per_instruction_results(monkeypatch, tmp_path):
    benchmark, generation_result = _ifbench_case(monkeypatch, tmp_path)

    scored_result = benchmark.evaluate_responses(generation_result)
    records = canonicalize_samples("IFBench", benchmark.to_samples(generation_result, scored_result))

    assert scored_result["strict_prompt_accuracy"] == 0.5
    assert scored_result["loose_prompt_accuracy"] == 0.5
    assert [record["strict_instruction_pass"] for record in records] == [[True], [False]]
    assert [record["loose_instruction_pass"] for record in records] == [[True], [False]]
    assert [record["strict_instruction_accuracy"] for record in records] == [1.0, 0.0]
    assert [record["resps"] for record in records] == [[["NoWhitespace"]], [["two words"]]]
    assert all("strict_instruction_pass" not in record["doc"] for record in records)


@pytest.mark.parametrize("task_name", ["ifeval", "ifeval_ca", "ifeval_es", "leaderboard_ifeval"])
def test_ifeval_native_instruction_metrics_preserve_flattened_accuracy(task_name):
    samples = [
        {
            "metrics": [
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            ],
            "prompt_level_strict_acc": False,
            "prompt_level_loose_acc": True,
            "inst_level_strict_acc": [True, False, False],
            "inst_level_loose_acc": [True, True, True],
        },
        {
            "metrics": [
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            ],
            "prompt_level_strict_acc": True,
            "prompt_level_loose_acc": True,
            "inst_level_strict_acc": [True],
            "inst_level_loose_acc": [True],
        },
    ]

    records = canonicalize_samples(task_name, samples)

    assert [record["inst_level_strict_acc"] for record in records] == [1 / 3, 1.0]
    assert [record["inst_level_loose_acc"] for record in records] == [1.0, 1.0]
    assert [record["strict_instruction_pass"] for record in records] == [[True, False, False], [True]]
    assert [record["loose_instruction_pass"] for record in records] == [[True, True, True], [True]]
    assert sum(map(sum, (record["strict_instruction_pass"] for record in records))) / sum(
        len(record["strict_instruction_pass"]) for record in records
    ) == 0.5


def test_other_tasks_reject_list_valued_metrics():
    with pytest.raises(SampleMetricsError, match="must be a finite number"):
        canonicalize_samples("other_task", [{"metrics": ["accuracy"], "accuracy": [True, False]}])


def test_repeated_accuracy_serializes_one_sample_per_trial():
    benchmark = _load_benchmark("GPQADiamond")
    generation_result = {
        "examples": [
            {
                "Question": "Choose A",
                "answer": "A",
                "model_outputs": ["A", "B", "A"],
                "model_answers": ["A", "B", "A"],
            }
        ]
    }

    scored_result = benchmark.evaluate_responses(generation_result)
    records = canonicalize_samples("GPQADiamond", benchmark.to_samples(generation_result, scored_result))

    assert [record["sample_repeat"] for record in records] == [0, 1, 2]
    assert [record["resps"] for record in records] == [[["A"]], [["B"]], [["A"]]]
    assert [record["accuracy"] for record in records] == [1.0, 0.0, 1.0]


def test_ifbench_scores_and_logs_samples_through_evaluation_driver(monkeypatch, tmp_path):
    benchmark, generation_result = _ifbench_case(monkeypatch, tmp_path)
    monkeypatch.setattr(benchmark, "generate_responses", lambda _model: generation_result)
    custom_tasks = SimpleNamespace(tasks={"IFBench": benchmark}, get_benchmark=lambda _task: benchmark)

    result = evaluate(
        lm=SimpleNamespace(rank=0, world_size=1),
        task_manager=custom_tasks,
        pretrain_task_manager=SimpleNamespace(all_tasks={}),
        task_list=["IFBench"],
        task_routes={"IFBench": CHAT_BENCHMARK_ROUTE},
        batch_sizes_list=[1],
        args=Namespace(model="local-chat-completions", log_samples=True),
    )

    assert result["results"]["IFBench"]["loose_prompt_accuracy"] == 0.5
    assert len(result["samples"]["IFBench"]) == 2
    assert result["task_outcomes"]["IFBench"]["status"] == "succeeded"


def test_a_sample_set_missing_its_metrics_is_rejected_at_the_serialization_boundary():
    with pytest.raises(SampleMetricsError, match="persists no per-sample metrics"):
        canonicalize_samples("MATH500", [{"doc_id": 0, "resps": [["4"]]}])


def test_a_recorded_metric_may_not_shadow_a_reserved_record_field():
    with pytest.raises(SampleMetricsError, match="shadows a reserved sample-record field"):
        record_sample_metrics({}, target=1.0)


def test_recorded_booleans_and_partial_credit_reach_the_record_as_floats():
    example: dict[str, Any] = {}
    record_sample_metrics(example, accuracy=True, prefix_hit=0.25)

    fields = sample_metric_fields(example)

    assert set(fields[SAMPLE_METRICS_FIELD]) == {"accuracy", "prefix_hit"}
    assert (fields["accuracy"], fields["prefix_hit"]) == (1.0, 0.25)


def test_a_listed_metric_without_a_value_is_rejected():
    with pytest.raises(SampleMetricsError, match="carries no value"):
        validate_sample_metrics("MATH500", [{"metrics": ["accuracy"]}])
