import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import boxed_math


@pytest.mark.parametrize("answer", ["25", r"\frac{50}{2}"])
def test_boxed_math_accepts_equivalent_final_answers(answer):
    spec = EvalSpec(task="AIME24", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("boxed_math"), target="25")

    assert boxed_math(spec, [rf"Work. \boxed{{{answer}}}"]).metrics["accuracy"] == 1.0


def test_boxed_math_rejects_an_unboxed_answer():
    spec = EvalSpec(task="AIME24", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("boxed_math"), target="25")

    assert boxed_math(spec, ["The answer is 25"]).metrics["accuracy"] == 0.0
