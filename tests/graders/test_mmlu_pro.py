import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import mmlu_choice


def test_mmlu_choice_uses_the_final_answer_marker():
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("mmlu_choice"), target="C")

    grade = mmlu_choice(spec, ["Earlier A was plausible. Answer: C"])

    assert grade.metrics["accuracy"] == 1.0
    assert grade.audit["filtered_resps"] == ["C"]


@pytest.mark.parametrize("response,answer", [
    ("B. This is a valid explanation.", "B"),
    ("A was plausible.\nB. This is a valid explanation.", "B"),
    ("b. This is a valid explanation.", None),
    ("The ANSWER IS (c). Later mention B.", "C"),
    ("aNsWeR: d. Later mention B.", "D"),
])
def test_mmlu_choice_preserves_original_case_sensitive_final_letter(response, answer):
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                    requests=(), grader=GraderSpec("mmlu_choice"), target=answer or "B")

    grade = mmlu_choice(spec, [response])

    assert grade.audit["filtered_resps"] == [answer]
    assert grade.metrics["accuracy"] == float(answer is not None)
    wrong = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0,
                     requests=(), grader=GraderSpec("mmlu_choice"), target="J")
    assert mmlu_choice(wrong, [response]).metrics["accuracy"] == 0.0
