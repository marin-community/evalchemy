"""Regression coverage for GSM8K's flexible answer filter."""

import pytest

from eval import robust_api  # noqa: F401 - registers the task filter at CLI startup
from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble


def _flexible_extract(response: str) -> tuple[Instance, str]:
    instance = Instance("generate_until", {}, (), 0)
    instance.resps = [response]
    pipeline = build_filter_ensemble(
        "flexible-extract",
        [
            ("gsm8k_flexible_extract", {}),
            ("take_first", {}),
        ],
    )
    pipeline.apply([instance])
    return instance, instance.filtered_resps["flexible-extract"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("$$\n\\boxed{18}\n$$", "18"),
        ("$18", "$18"),
        ("Answer: 18.", "18"),
        ("The calculation used 9 and 2. Final answer: 18.", "18"),
        ("\\boxed{-3.5}", "-3.5"),
        ("\\boxed{1,200}", "1,200"),
        ("\\boxed{18}. The calculation used 9 and 2.", "18"),
        ("$$", "[invalid]"),
    ],
)
def test_flexible_extract_uses_final_answer_syntax_before_numeric_fallback(response, expected):
    instance, selected = _flexible_extract(response)

    assert instance.resps == [response]
    assert selected == expected
