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


def test_case_instructions_cannot_ignore_cased_nonalphabetic_characters(tmp_path):
    from eval.chat_benchmarks.IFEval.instructions import CapitalLettersEnglishChecker

    source = CapitalLettersEnglishChecker("change_case:english_capital")
    source.build_description()
    english = "WAN WAN IS A POWERFUL AND CUNNING VILLAIN IN THE LEGEND OF THE SWORD AND THE FAIRY."
    assert source.check_following(english)
    assert not source.check_following(english + "ⅰ")
    assert source.check_following(english + "中")
    responses = [english, english + "ⅰ", english + "中", "", "123"]
    result = score(
        tmp_path,
        [row(str(index), value, ["change_case:english_capital"], [{}]) for index, value in enumerate(responses)]
        + [row("5", english.lower(), ["language:response_language"], [{"language": "en"}])],
    )
    assert result["per_prompt_follow_rate"] == {
        "0": {"strict": 1.0, "loose": 1.0},
        "1": {"strict": 0.0, "loose": 0.0},
        "2": {"strict": 0.0, "loose": 0.0},
        "3": {"strict": 0.0, "loose": 0.0},
        "4": {"strict": 0.0, "loose": 0.0},
        "5": {"strict": 1.0, "loose": 1.0},
    }


def test_ifbench_cached_float_bounds_and_empty_derived_options(tmp_path):
    path = tmp_path / "ifbench.jsonl"
    path.write_text(
        json.dumps(row("count", "one two", ["count:word_count_range"], [{"min_words": 2.0, "max_words": 3.0}])) + "\n"
    )
    result = evaluate_accuracy(path, "IFBench")
    assert result["strict_prompt_accuracy"] == 1.0
    path.write_text(json.dumps(row("options", "!!!", ["format:options"], [{"options": "/"}])) + "\n")
    with pytest.raises(InvalidTask):
        evaluate_accuracy(path, "IFBench")


@pytest.mark.parametrize("family,instruction", [("IFEval", "punctuation:no_comma"), ("IFBench", "format:newline")])
def test_blank_candidate_is_zero_without_invalidating_instruction_task(tmp_path, family, instruction):
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps(row("empty", " \n", [instruction], [{}])) + "\n")
    result = evaluate_accuracy(path, family)
    metric = "prompt-level" if family == "IFEval" else "strict_prompt_accuracy"
    assert result[metric] == 0.0
    with path.open("a") as handle:
        handle.write(json.dumps(row("positive", "hello", [instruction], [{}])) + "\n")
    mixed = evaluate_accuracy(path, family)
    assert mixed[metric] == 0.5
    if family == "IFEval":
        assert mixed["per_prompt_follow_rate"] == {
            "empty": {"strict": 0.0, "loose": 0.0},
            "positive": {"strict": 1.0, "loose": 1.0},
        }
    else:
        assert [item["strict_instruction_pass"] for item in mixed["per_prompt_outcomes"]] == [[False], [True]]
    invalid_name, invalid_args = (
        ("keywords:existence", {"keywords": []}) if family == "IFEval" else ("format:options", {"options": ""})
    )
    path.write_text(json.dumps(row("invalid", "", [invalid_name], [invalid_args])) + "\n")
    with pytest.raises(InvalidTask):
        evaluate_accuracy(path, family)


def test_ifbench_precision_zero_reference_and_invalid_metadata_precedence(tmp_path):
    path = tmp_path / "precision.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                row(
                    str(index),
                    candidate,
                    ["ratio:overlap"],
                    [
                        {
                            "reference_text": reference,
                            "percentage": 0,
                        }
                    ],
                )
            )
            for index, (reference, candidate) in enumerate([("abc", "xyz"), ("", "xyz"), ("abc", "")])
        )
        + "\n"
    )
    result = evaluate_accuracy(path, "IFBench")
    assert [item["strict_instruction_pass"] for item in result["per_prompt_outcomes"]] == [[True], [True], [False]]
    for reference, percentage in [(None, 0), ("abc", float("nan"))]:
        path.write_text(
            json.dumps(
                row(
                    "invalid",
                    "",
                    ["ratio:overlap"],
                    [
                        {
                            "reference_text": reference,
                            "percentage": percentage,
                        }
                    ],
                )
            )
            + "\n"
        )
        with pytest.raises(InvalidTask):
            evaluate_accuracy(path, "IFBench")


def test_ifbench_rejects_impossible_dates_and_incomplete_csv_after_valid_row(tmp_path):
    from eval.chat_benchmarks.IFBench.instructions import DateFormatListChecker, SpecialCharacterCSVChecker

    dates = DateFormatListChecker("custom:date_format_list")
    dates.build_description()
    malformed_dates = ["1800-00-01", "1800-01-00", "1800-02-29"]
    assert all(dates.check_following(value) for value in malformed_dates)
    csv_source = SpecialCharacterCSVChecker("custom:csv_special_character")
    csv_source.build_description()
    csv_text = 'ProductID,Category,Brand,Price,Stock\n1,"a,b",C,1,2\n' + "1,2,3,4,5\n" * 12 + "missing,columns\n"
    assert csv_source.check_following(csv_text)
    cases = [row(str(index), value, ["custom:date_format_list"], [{}]) for index, value in enumerate(malformed_dates)]
    cases += [
        row("csv", csv_text, ["custom:csv_special_character"], [{}]),
        row("valid", "1804-02-29", ["custom:date_format_list"], [{}]),
    ]
    path = tmp_path / "invalid-structure.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in cases) + "\n")
    result = evaluate_accuracy(path, "IFBench")
    assert [item["strict_instruction_pass"] for item in result["per_prompt_outcomes"]] == [[False]] * 4 + [[True]]


def test_ifbench_case_and_nesting_require_the_requested_structure(tmp_path):
    from eval.chat_benchmarks.IFBench.instructions import NestedQuotesChecker, TitleCaseChecker

    title = TitleCaseChecker("format:title_case")
    title.build_description()
    assert title.check_following("hELlo")
    quotes = NestedQuotesChecker("format:quotes")
    quotes.build_description()
    incomplete = "".join(['"', "'", '"', "'", "text", "'", '"', "'"])
    assert quotes.check_following(incomplete)
    closed = incomplete + '"'
    cases = [
        row("bad-case", "hELlo", ["format:title_case"], [{}]),
        row("case", "Hello World", ["format:title_case"], [{}]),
        row("open", incomplete, ["format:quotes"], [{}]),
        row("closed", closed, ["format:quotes"], [{}]),
        row("shallow", "\"one\" 'two'", ["format:quotes"], [{}]),
    ]
    path = tmp_path / "structure.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in cases) + "\n")
    result = evaluate_accuracy(path, "IFBench")
    assert [item["strict_instruction_pass"] for item in result["per_prompt_outcomes"]] == [
        [False],
        [True],
        [False],
        [True],
        [False],
    ]


def test_ifbench_distinct_words_and_bullets_cannot_pass_vacuously(tmp_path):
    from eval.chat_benchmarks.IFBench.instructions import (
        CharacterCountUniqueWordsChecker,
        ConjunctionCountChecker,
        SubBulletPointsChecker,
    )

    conjunctions = ConjunctionCountChecker("count:conjunctions")
    conjunctions.build_description(small_n=3)
    assert conjunctions.check_following("and AND And")
    sentences = CharacterCountUniqueWordsChecker("ratio:sentence_words")
    sentences.build_description()
    assert sentences.check_following("Cat. Cat. Cat.")
    bullets = SubBulletPointsChecker("format:sub-bullets")
    bullets.build_description()
    assert bullets.check_following("Plain prose")
    assert bullets.check_following("* main - nested")
    cases = [
        row("repeated-conjunction", "and AND And", ["count:conjunctions"], [{"small_n": 3}]),
        row("distinct-conjunction", "and but for", ["count:conjunctions"], [{"small_n": 3}]),
        row("repeated-sentence", "Cat. Cat. Cat.", ["ratio:sentence_words"], [{}]),
        row("distinct-sentence", "Cat. Dog. Pig.", ["ratio:sentence_words"], [{}]),
        row("no-bullets", "Plain prose", ["format:sub-bullets"], [{}]),
        row("inline-bullets", "* main - nested", ["format:sub-bullets"], [{}]),
        row("nested-bullets", "* main\n- nested", ["format:sub-bullets"], [{}]),
    ]
    path = tmp_path / "distinct-structure.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in cases) + "\n")
    result = evaluate_accuracy(path, "IFBench")
    assert [item["strict_instruction_pass"] for item in result["per_prompt_outcomes"]] == [
        [False],
        [True],
        [False],
        [True],
        [False],
        [False],
        [True],
    ]
