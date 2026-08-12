from eval.chat_benchmarks.NUPA.scorer import (
    FLOAT,
    FRACTION,
    INTEGER,
    SCIENTIFIC,
    extract_answer,
    length_bucket,
    normalize_answer,
    score_prediction,
)


def test_extract_answer_accepts_official_api_answer_marker():
    assert extract_answer("The answer is 9.9, because 9.9 is larger.", FLOAT) == "9.9"


def test_extract_answer_does_not_search_explanatory_prose():
    assert extract_answer("I think the answer is 9.9", FLOAT) is None


def test_exact_match_float_nupa_trap():
    score = score_prediction("9.9", "9.9", FLOAT)
    assert score.exact_match == 1.0
    assert score.digit_match == 1.0
    assert score.dlength == 0.0


def test_exact_match_rejects_9_11_for_9_9():
    score = score_prediction("9.11", "9.9", FLOAT)
    assert score.exact_match == 0.0
    # integer part matches; first decimal digit does not.
    assert score.digit_match == 0.5
    assert score.dlength == 1.0


def test_exact_match_preserves_leading_zeroes_and_does_not_ignore_commas():
    assert score_prediction("001234", "1234", INTEGER).exact_match == 0.0
    assert score_prediction("1,234", "1234", INTEGER).exact_match == 0.0


def test_exact_match_preserves_float_representation():
    assert score_prediction("9.90", "9.9", FLOAT).exact_match == 0.0
    assert score_prediction("9", "9.0", FLOAT).format_valid == 0.0


def test_fraction_scoring_handles_components():
    score = score_prediction("3/2", "3/2", FRACTION)
    assert score.exact_match == 1.0
    assert score.digit_match == 1.0


def test_scientific_notation_normalization():
    assert normalize_answer("05.040e+04", SCIENTIFIC) == "05.040e04"
    score = score_prediction("5.04e4", "5.04e4", SCIENTIFIC)
    assert score.exact_match == 1.0


def test_invalid_format_is_not_format_valid():
    score = score_prediction("one half", "1/2", FRACTION)
    assert score.exact_match == 0.0
    assert score.digit_match == 0.0
    assert score.format_valid == 0.0


def test_dlength_compares_total_digit_count():
    score = score_prediction("123/456", "12/3456", FRACTION)
    assert score.dlength == 0.0


def test_length_bucket_boundaries():
    assert length_bucket(4, max_digit=20) == "S"
    assert length_bucket(8, max_digit=20) == "M"
    assert length_bucket(14, max_digit=20) == "L"
    assert length_bucket(15, max_digit=20) == "XL"
    assert length_bucket(10, max_digit=100) == "S"
    assert length_bucket(20, max_digit=100) == "M"
    assert length_bucket(60, max_digit=100) == "L"
    assert length_bucket(61, max_digit=100) == "XL"
