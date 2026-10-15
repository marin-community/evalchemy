from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import mmlu_choice


def test_mmlu_choice_uses_the_final_answer_marker():
    spec = EvalSpec(task="MMLUPro", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("mmlu_choice"), target="C")

    grade = mmlu_choice(spec, ["Earlier A was plausible. Answer: C"])

    assert grade.metrics["accuracy"] == 1.0
    assert grade.audit["filtered_resps"] == ["C"]
