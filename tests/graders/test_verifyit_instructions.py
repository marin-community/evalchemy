"""Source instruction contracts execute through real IFEval and Script graders."""

import json

import pytest

from eval.graders.verifyit_instructions import evaluate_accuracy
from verifyit.grade import InvalidTask


def score(tmp_path, rows):
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return evaluate_accuracy(path, "IFEval")


def row(prompt, response, ids=None, kwargs=None):
    return {
        "key": prompt,
        "prompt": prompt,
        "response": response,
        "instruction_id_list": ["punctuation:no_comma"] if ids is None else ids,
        "kwargs": [{}] if kwargs is None else kwargs,
    }


def test_strict_and_loose_source_protocols_remain_distinct(tmp_path):
    result = score(tmp_path, [row("one", "header,\nhello\nfooter,"), row("two", "wrong, answer")])
    assert result["per_prompt_follow_rate"] == {
        "one": {"strict": 0.0, "loose": 1.0},
        "two": {"strict": 0.0, "loose": 0.0},
    }
    assert result["prompt-level"] == 0.5
    assert result["instruction-level"] == 0.5


def test_language_detection_failure_cannot_award_credit(tmp_path):
    result = score(
        tmp_path,
        [
            row("language", "123", ["language:response_language"], [{"language": "en"}]),
            row(
                "valid",
                "The quick brown fox jumps over the lazy dog and then returns home.",
                ["language:response_language"],
                [{"language": "en"}],
            ),
        ],
    )
    assert result["per_prompt_follow_rate"]["language"] == {"strict": 0.0, "loose": 0.0}
    assert result["per_prompt_follow_rate"]["valid"] == {"strict": 1.0, "loose": 1.0}


@pytest.mark.parametrize(
    "bad",
    [
        row("invalid", "hello", [], []),
        row("invalid", "hello", ["unknown:predicate"], [{}]),
        row("invalid", "hello", ["keywords:existence"], []),
        row("valid", "overwritten"),
    ],
)
def test_invalid_instruction_batch_returns_no_partial_metrics(tmp_path, bad):
    with pytest.raises(InvalidTask):
        score(tmp_path, [row("valid", "hello"), bad])


@pytest.mark.parametrize(
    "fault", ["missing_result", "missing_flags", "nonfinite_metric", "nonfinite_rng", "inconsistent_aggregate"]
)
def test_malformed_producer_result_does_not_change_host_rng(tmp_path, monkeypatch, fault):
    import random

    from eval.graders import verifyit_instructions
    from verifyit.grade import scored

    before = random.getstate()
    proposed_state = list(random.Random(54321).getstate())
    result = {
        "prompt-level": 1.0,
        "instruction-level": 1.0,
        "per_prompt_follow_rate": {"one": {"strict": 1.0, "loose": 1.0}},
    }
    if fault == "missing_result":
        result = None
    elif fault == "missing_flags":
        result["per_prompt_follow_rate"] = {}
    elif fault == "nonfinite_metric":
        result["prompt-level"] = float("nan")
    elif fault == "inconsistent_aggregate":
        result["per_prompt_follow_rate"]["one"] = {"strict": 0.0, "loose": 0.0}
    else:
        proposed_state[2] = float("nan")
    monkeypatch.setattr(
        verifyit_instructions, "run", lambda *args: scored(1.0, source_result=result, random_state=proposed_state)
    )
    with pytest.raises((RuntimeError, InvalidTask)):
        score(tmp_path, [row("one", "hello")])
    assert random.getstate() == before


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("keywords:existence", {"keywords": []}),
        ("keywords:existence", {"keywords": [""]}),
        ("keywords:existence", {"keywords": [".*"]}),
        ("length_constraints:number_words", {"num_words": 0, "relation": "at least"}),
    ],
)
def test_vacuous_instruction_reference_cannot_score_wrong_candidate(tmp_path, name, arguments):
    with pytest.raises(InvalidTask):
        score(tmp_path, [row("valid", "hello"), row("invalid", "wrong", [name], [arguments])])


@pytest.mark.parametrize("fault", ["prompt", "instruction", "type"])
def test_ifbench_aggregate_must_match_returned_flags_before_rng_commit(tmp_path, monkeypatch, fault):
    import random

    from eval.graders import verifyit_instructions
    from verifyit.grade import scored

    before = random.getstate()
    result = {
        "strict_prompt_accuracy": 0.0,
        "strict_instruction_accuracy": 0.0,
        "strict_per_type": {"format": 0.0},
        "loose_prompt_accuracy": 0.0,
        "loose_instruction_accuracy": 0.0,
        "loose_per_type": {"format": 0.0},
        "per_prompt_outcomes": [
            {"prompt": "one", "strict_instruction_pass": [False], "loose_instruction_pass": [False]}
        ],
    }
    if fault == "type":
        result["loose_per_type"]["format"] = 1.0
    else:
        result["loose_" + fault + "_accuracy"] = 1.0
    monkeypatch.setattr(
        verifyit_instructions,
        "run",
        lambda *args: scored(1.0, source_result=result, random_state=random.Random(54321).getstate()),
    )
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps(row("one", "wrong answer", ["format:no_whitespace"], [{}])) + "\n")
    with pytest.raises(RuntimeError, match="Inconsistent IFBench"):
        evaluate_accuracy(path, "IFBench")
    assert random.getstate() == before


def test_json_instruction_rejects_ambiguous_and_recursive_candidates(tmp_path):
    from eval.chat_benchmarks.IFEval.instructions import JsonFormat

    source = JsonFormat("detectable_format:json_format")
    source.build_description()
    ambiguous = ['{"answer": 1, "answer": 2}', '{"answer": NaN}']
    assert all(source.check_following(value) for value in ambiguous)
    responses = ['{"answer": 1}', *ambiguous, "[" * 1200 + "0" + "]" * 1200]
    result = score(
        tmp_path,
        [row(str(index), value, ["detectable_format:json_format"], [{}]) for index, value in enumerate(responses)],
    )
    assert result["per_prompt_follow_rate"] == {
        "0": {"strict": 1.0, "loose": 1.0},
        "1": {"strict": 0.0, "loose": 0.0},
        "2": {"strict": 0.0, "loose": 0.0},
        "3": {"strict": 0.0, "loose": 0.0},
    }
