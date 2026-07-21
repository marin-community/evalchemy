"""Regression tests for the DROP answer-extraction filter (marin-community/evalchemy#31).

Background
----------
``drop`` is scored as a ``generate_until`` task with the strict DROP token-F1 metric
(``lm_eval.tasks.drop.utils.process_results``). When a model is run WITHOUT a chat
template (``local-completions`` — required so the MC / loglikelihood tasks in the same
run behave), the completion is a long, verbose continuation rather than a short answer
span. Token-F1 divides by the size of the *predicted* bag of tokens, so a verbose
completion that literally contains the correct answer still collapses to f1 ~= 0. That
is exactly what happened grid-wide (all six A3B MoE models: f1 ~= 0.002 despite gsm8k
0.68-0.95).

The fix is an answer-extraction filter that pulls the short answer out of the verbose
completion BEFORE token-F1 scoring. These tests pin the real scoring path
(``lm_eval``'s ``process_results`` / ``get_metrics``) and the real override wiring
(``lm_eval``'s ``TaskManager`` include-path precedence), so they exercise the same code
that runs on the cluster.
"""

import sys
from pathlib import Path

# The evalchemy `eval` package is copied into site-packages, not editable-installed, so
# put this worktree first on the path to import the in-tree fix under test.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lm_eval.tasks.drop.utils import process_results  # noqa: E402

from eval.lm_eval_tasks.drop.utils import drop_answer_extraction_filter, extract_drop_short_answer  # noqa: E402

# A realistic no-template completion: the model restates the passage/question and only
# incidentally contains the gold number ("42"). This is what the DROP metric actually
# receives when there is no chat template.
VERBOSE_COMPLETION = (
    "Looking at the passage we can see that the home team and the away team both "
    "played several drives and the question asks how many total points were scored "
    "in the game so we add up all of the touchdowns and field goals mentioned to "
    "arrive at a grand total of 42 points across the entire contest as described"
)

GOLD_NUMBER_DOC = {"answers": [("42",)]}


def _f1(doc, results):
    """Run the REAL DROP scoring path and return the f1 scalar."""
    return process_results(doc, results)["f1"]


def _filter_one(completion):
    """Apply the extraction filter to a single completion (per-doc, single-sample)."""
    return drop_answer_extraction_filter([[completion]], [{}])[0]


# --------------------------------------------------------------------------------------
# The bug (issue #31): raw verbose completions collapse to ~0 f1.
# --------------------------------------------------------------------------------------
def test_verbose_completion_collapses_on_raw_path():
    """Reproduce the bug: the raw verbose completion collapses to ~0 f1.

    The completion contains the correct answer ("42") but scores near zero because
    precision is diluted by the verbose surrounding tokens. This documents WHY the
    extraction filter is needed and guards against anyone "fixing" it by weakening the
    metric itself.
    """
    f1 = _f1(GOLD_NUMBER_DOC, [VERBOSE_COMPLETION])
    assert f1 < 0.1, f"expected collapsed f1 (<0.1) on raw verbose completion, got {f1}"


# --------------------------------------------------------------------------------------
# The fix: the extraction filter recovers the correct answer.
# --------------------------------------------------------------------------------------
def test_filter_recovers_verbose_completion():
    """The extraction filter recovers the correct answer to a real (f1 == 1) score."""
    f1 = _f1(GOLD_NUMBER_DOC, _filter_one(VERBOSE_COMPLETION))
    assert f1 == 1.0, f"expected f1==1.0 after extraction, got {f1}"


def test_extract_marker_mid_text():
    """An explicit answer marker anywhere in the text is honored (answer mid-text)."""
    doc = {"answers": [("27",)]}
    comp = "After careful analysis, the answer is 27 touchdowns scored in the game."
    assert extract_drop_short_answer(comp) == "27"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_multiple_numbers_takes_final():
    """With several numbers and no marker, the final (bottom-line) number is taken."""
    doc = {"answers": [("28",)]}
    comp = "There were 3 field goals and 4 touchdowns, but the final total was 28"
    assert extract_drop_short_answer(comp) == "28"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_marker_wins_over_earlier_numbers():
    """The LAST answer marker wins even when intermediate numbers appear first."""
    doc = {"answers": [("15",)]}
    comp = "we compute 10 plus 5 which means the answer is 15"
    assert extract_drop_short_answer(comp) == "15"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_full_date_answer():
    """A day/month/year date span is extracted intact (matches DROP gold order)."""
    doc = {"answers": [("24 May 1993",)]}
    comp = "The event happened on 24 May 1993 according to the passage"
    assert extract_drop_short_answer(comp) == "24 May 1993"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_month_year_date_answer():
    """A month/year date (no day) is extracted and scores correctly."""
    doc = {"answers": [("December 1941",)]}
    comp = "It occurred in December 1941 during the war, per the text."
    assert extract_drop_short_answer(comp) == "December 1941"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_decimal_and_negative():
    """Signed / decimal figures survive extraction."""
    assert extract_drop_short_answer("the difference came out to -3.5 in the end") == "-3.5"


def test_exact_short_answer_is_idempotent():
    """An already-short correct answer is unchanged and still scores 1.0."""
    assert extract_drop_short_answer("42") == "42"
    assert _f1(GOLD_NUMBER_DOC, _filter_one("42")) == 1.0


# --------------------------------------------------------------------------------------
# The metric must NOT be weakened: genuinely-wrong completions still score ~0.
# --------------------------------------------------------------------------------------
def test_wrong_numeric_answer_stays_zero():
    """A verbose completion committing to the WRONG number still scores 0."""
    comp = "the total was 17 points scored by both teams combined in the game"
    assert extract_drop_short_answer(comp) == "17"
    assert _f1(GOLD_NUMBER_DOC, _filter_one(comp)) == 0.0


def test_no_answer_present_does_not_fabricate():
    """When no number/date/marker is present, extraction does not invent the gold."""
    comp = "the home team ultimately won the game after a hard fought contest"
    # Falls back to the (verbose) text -> still collapses; we never fabricate "42".
    assert _f1(GOLD_NUMBER_DOC, _filter_one(comp)) < 0.1


# --------------------------------------------------------------------------------------
# Filter plumbing + robustness.
# --------------------------------------------------------------------------------------
def test_filter_preserves_shape_over_docs_and_samples():
    """The filter maps list[list[str]] -> list[list[str]] element-wise."""
    resps = [
        ["the answer is 7", "and 7 again"],
        ["nothing here about 12", "the answer is 12"],
    ]
    out = drop_answer_extraction_filter(resps, [{}, {}])
    assert out == [["7", "7"], ["12", "12"]]


def test_filter_handles_non_string_and_empty():
    """Non-string / empty responses degrade to '' rather than crashing."""
    out = drop_answer_extraction_filter([[None, "", "  "]], [{}])
    assert out == [["", "", ""]]


# --------------------------------------------------------------------------------------
# Wiring: the override YAML actually takes precedence over upstream `drop`.
# --------------------------------------------------------------------------------------
def test_override_yaml_takes_precedence():
    """The evalchemy include dir overrides the packaged `drop` task and adds the filter."""
    from lm_eval.tasks import TaskManager

    from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR

    tm = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])
    entry = tm.task_index.get("drop")
    assert entry is not None and entry.yaml_path is not None
    # The resolved `drop` config must be OUR override file, not the packaged one.
    assert Path(entry.yaml_path).resolve() == (Path(DEFAULT_LM_EVAL_INCLUDE_DIR) / "drop" / "drop.yaml").resolve()
