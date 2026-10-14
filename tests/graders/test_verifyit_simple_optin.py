"""Exact and numeric cutovers preserve defaults and reject bad batches before scoring."""

import copy

import pytest

from eval.chat_benchmarks.AIW.eval_instruct import AIWBenchmark
from eval.chat_benchmarks.AMC23.eval_instruct import AMC23Benchmark
from eval.chat_benchmarks.GSM8KPerturbed.eval_instruct import GSM8KPerturbedBenchmark
from verifyit.grade import InvalidTask


@pytest.mark.parametrize(
    ("benchmark", "examples", "reference"),
    [
        (AIWBenchmark, [{"id": 577, "right_answer": "2", "model_answer": "2"}], "right_answer"),
        (AMC23Benchmark, [{"id": 0, "answer": "2", "model_answers": ["2"] * 10}], "answer"),
        (
            GSM8KPerturbedBenchmark,
            [{"id": "one", "task": "gsm8k-clean", "answer": "2", "output": "The answer is 2"}],
            "answer",
        ),
    ],
)
@pytest.mark.parametrize("invalid_reference", ["", None, "nan", True])
def test_cutover_matches_default_and_bad_later_reference_preserves_input(
    benchmark, examples, reference, invalid_reference
):
    options = {"n_trials": 1} if benchmark is AIWBenchmark else {}
    source, cutover = {"examples": copy.deepcopy(examples)}, {"examples": copy.deepcopy(examples)}
    assert benchmark(**options).evaluate_responses(source) == benchmark(
        verifyit_enabled=True, **options
    ).evaluate_responses(cutover)
    assert source == cutover
    bad = copy.deepcopy(examples[0])
    bad["id"], bad[reference] = "later", invalid_reference
    values = {"examples": copy.deepcopy(examples) + [bad]}
    original = copy.deepcopy(values)
    with pytest.raises(InvalidTask):
        benchmark(verifyit_enabled=True, **options).evaluate_responses(values)
    assert values == original


def test_gsm_default_missing_candidate_short_circuits_invalid_reference():
    values = {"examples": [{"id": "one", "task": "gsm8k-clean", "answer": None, "output": "No answer"}]}
    GSM8KPerturbedBenchmark().evaluate_responses(values)
    assert values["examples"][0]["correct"] is False
    with pytest.raises(InvalidTask):
        GSM8KPerturbedBenchmark(verifyit_enabled=True).evaluate_responses(copy.deepcopy(values))
