import pytest

from eval.chat_benchmarks.NUPA.scorer import (
    FLOAT,
    FRACTION,
    INTEGER,
    SCIENTIFIC,
    extract_answer,
    length_bucket,
    score_prediction,
)


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        ("The answer is 9.9, because 9.9 is larger.", FLOAT, "9.9"),
        ("I think the answer is 9.9", FLOAT, None),
        ("So the answer is 123", INTEGER, "123"),
        ("5.04E+04", SCIENTIFIC, "5.04e04"),
    ],
)
def test_extract_answer_follows_direct_answer_protocol(prediction, answer_format, expected):
    assert extract_answer(prediction, answer_format) == expected


def test_stored_step92_answers_are_extracted_without_changing_exact_match():
    fraction_output = (
        "First, I need to calculate the result of dividing 56 by 84616. This can be written as the fraction "
        "\\(\\frac{56}{84616}\\).\n\n"
        "Next, I'll simplify this fraction by finding the greatest common divisor (GCD) of 56 and 84616.\n\n"
        "The GCD of 56 and 84616 is 56.\n\n"
        "Now, I'll divide both the numerator and the denominator by 56:\n\n"
        "\\(\\frac{56 \\div 56}{84616 \\div 56} = \\frac{1}{1511}\\)\n\n"
        "The simplified fraction is \\(\\boxed{1/1511}\\). "
    )
    rounded_float_output = (
        "The result is calculated as follows:  \n"
        "426105174.011715486 + 19.07129229308 = 426105193.08300778 "
    )

    assert extract_answer(fraction_output, FRACTION) == "1/1511"
    assert score_prediction(fraction_output, "1/1511", FRACTION).exact_match == 1.0
    assert extract_answer(rounded_float_output, FLOAT) == "426105193.08300778"
    assert score_prediction(rounded_float_output, "426105193.08300777908", FLOAT).exact_match == 0.0


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        (r"<|start_think|>Maybe \boxed{9}<|end_think|>\(\boxed{1/1511}\).", FRACTION, "1/1511"),
        (r"<|start_think|>Maybe 9<|end_think|>\(1/1511\).", FRACTION, "1/1511"),
        ("<|start_think|>Maybe 9<|end_think|>42", INTEGER, "42"),
        (r"<|start_think|>\boxed{9}", INTEGER, None),
        (r"The answer is \boxed{9.90}.", FLOAT, "9.90"),
        ("So the answer is: 83564549", INTEGER, "83564549"),
        ("Therefore, the final answer is 2.547762432164e18.", SCIENTIFIC, "2.547762432164e18"),
        (r"\boxed{1,234}", INTEGER, None),
        (r"\boxed{9.90}", INTEGER, None),
        ("Result: 9.9\nI need to check that", FLOAT, None),
    ],
)
def test_extract_answer_uses_final_content(prediction, answer_format, expected):
    assert extract_answer(prediction, answer_format) == expected


def test_score_prediction_preserves_numeric_representation():
    exact = score_prediction("9.9", "9.9", FLOAT)
    wrong_digits = score_prediction("9.11", "9.9", FLOAT)
    extra_zero = score_prediction("9.90", "9.9", FLOAT)
    comma = score_prediction("1,234", "1234", INTEGER)

    assert exact.exact_match == 1.0
    assert exact.digit_match == 1.0
    assert exact.dlength == 0.0
    assert wrong_digits.exact_match == 0.0
    assert wrong_digits.digit_match == 0.5
    assert wrong_digits.dlength == 1.0
    assert extra_zero.exact_match == 0.0
    assert comma.exact_match == 0.0


def test_fraction_and_scientific_components_are_scored_independently():
    fraction = score_prediction("123/456", "12/3456", FRACTION)
    scientific = score_prediction("5.04e4", "5.04e4", SCIENTIFIC)

    assert fraction.dlength == 0.0
    assert scientific.exact_match == 1.0


def test_invalid_format_records_no_answer():
    score = score_prediction("one half", "1/2", FRACTION)

    assert score.exact_match == 0.0
    assert score.digit_match == 0.0
    assert score.format_valid == 0.0
    assert score.no_answer == 1.0


@pytest.mark.parametrize(
    ("digit", "max_digit", "expected"),
    [
        (4, 20, "S"),
        (8, 20, "M"),
        (14, 20, "L"),
        (15, 20, "XL"),
        (10, 100, "S"),
        (20, 100, "M"),
        (60, 100, "L"),
        (61, 100, "XL"),
    ],
)
def test_length_bucket_matches_the_paper_intervals(digit, max_digit, expected):
    assert length_bucket(digit, max_digit=max_digit) == expected
