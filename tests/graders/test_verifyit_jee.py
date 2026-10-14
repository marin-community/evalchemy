"""JEEBench grading uses raw outputs and aborts incomplete cohorts."""

import copy

import pytest

from eval.graders.verifyit_jee import grade_batch
from eval.contracts.failures import GradingBoundaryError
from verifyit.grade import Status


def test_jee_batch_uses_raw_outputs_for_partial_choice_and_numeric_credit():
    examples = [
        {"gold": "ACD", "type": "MCQ(multiple)", "model_outputs": [r"\boxed{CA}", r"\boxed{AB}", r"\boxed{ADCC}"],
         "model_answers": ["ACD", "ACD", "ACD"]},
        {"gold": "0", "type": "Numeric", "model_outputs": [r"\boxed{0.01}", r"\boxed{0.0100001}", None]},
    ]
    original = copy.deepcopy(examples)
    records = grade_batch(examples, 3)
    assert [[result["reward"] for result in row["verdicts"]] for row in records] == [[0.5, 0, 1], [1, 0, 0]]
    assert all(result["status"] == Status.SCORED for row in records for result in row["verdicts"])
    assert "reference_sha256" in records[0]
    assert "raw" not in records[0] and "prepared" not in records[0]
    assert "detail" not in records[0]["verdicts"][0]
    assert examples == original


@pytest.mark.parametrize("broken", [
    {"gold": "", "type": "MCQ", "model_outputs": [""]},
    {"gold": "nan", "type": "Numeric", "model_outputs": [""]},
])
def test_jee_batch_invalid_reference_aborts_after_positive_sample_without_mutation(broken):
    examples = [{"gold": "A", "type": "MCQ", "model_outputs": [r"\boxed{A}"]}, broken]
    original = copy.deepcopy(examples)
    with pytest.raises(GradingBoundaryError) as raised:
        grade_batch(examples, 1)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert raised.value.verdict["reward"] == 0
    assert examples == original


def test_jee_batch_missing_outputs_are_infrastructure_failure():
    examples = [{"gold": "A", "type": "MCQ", "model_answers": ["A"]}]
    with pytest.raises(GradingBoundaryError) as raised:
        grade_batch(examples, 1)
    assert raised.value.verdict["status"] == Status.INFRA_ERROR
    assert raised.value.verdict["reward"] == 0
    assert "score" not in examples[0]


def test_jee_trusted_reference_failure_precedes_missing_provider_outputs():
    examples = [{"gold": "A", "type": "MCQ"}, {"gold": "nan", "type": "Numeric", "model_outputs": [""]}]
    with pytest.raises(GradingBoundaryError) as raised:
        grade_batch(examples, 1)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert raised.value.verdict["reward"] == 0
    assert all("score" not in example for example in examples)
