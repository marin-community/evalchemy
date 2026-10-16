"""RepoBench retains the original label-first edit-similarity policy."""

from difflib import SequenceMatcher

from fuzzywuzzy import fuzz
import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.structured import code_line


@pytest.mark.parametrize(
    ("label", "completion", "expected_em", "expected_es"),
    [("tide", "diet", 0, 25), ("diet", "tide", 0, 50), ("same", "same", 100, 100), ("tide", "", 0, 0)],
)
def test_repobench_edit_similarity_preserves_label_first_backend_policy(monkeypatch, label, completion, expected_em, expected_es):
    # Exercise the real supported stdlib backend even when optional C extensions
    # are installed; its directional matching makes argument order observable.
    monkeypatch.setattr(fuzz, "SequenceMatcher", SequenceMatcher)
    spec = EvalSpec("RepoBench", SourceRef("fixture"), 0, 0, (), GraderSpec("code_line", {"language": "python"}), target=label)

    grade = code_line(spec, [completion])

    assert grade.metrics == {"EM": expected_em, "ES": expected_es}
