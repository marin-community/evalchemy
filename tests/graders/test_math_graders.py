"""Parity tests for the self-contained MATH graders in eval/graders.

lm-eval is a base dependency here, so every case is checked against the pinned
harness live rather than against recorded verdicts: the reference functions are
imported and run side by side with the port.

``data/math_grader_benchmark.jsonl`` holds 100 MATH test problems spanning all
seven subjects, each with a synthesized completion in the format its task's
prompt elicits, mixing verbatim, equivalently reformatted, wrong, and malformed
answers.

The single-case tests cover divergences a plausible fast path gets wrong, each
observed against the reference rather than invented.

The equivalence waivers (whitespace, ``\\dfrac``→``\\frac``,
``$``/``\\text{...}`` wrappers, and similar reformulations) are deliberate
divergences from the harness. The parity tests below scope them out, and the
``*_equivalences`` tests pin the new behavior instead.
"""

import json
import pathlib
import sys

import pytest
from lm_eval.tasks.hendrycks_math import utils as reference_hendrycks
from lm_eval.tasks.minerva_math import utils as reference_minerva

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import hendrycks_math, minerva_math  # noqa: E402

BENCHMARK = pathlib.Path(__file__).parent / "data" / "math_grader_benchmark.jsonl"


@pytest.fixture(scope="module")
def benchmark():
    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    assert len(records) == 100
    return records


def reference_hendrycks_grade(solution: str, reference_answer: str) -> float:
    """The harness ``process_results`` extraction and comparison, inlined."""
    indices = [pos for pos, char in enumerate(solution) if char == "$"]
    answer = solution if len(indices) <= 1 else solution[indices[0] + 1 : indices[-1]]
    return 1.0 if reference_hendrycks.is_equiv(answer, reference_answer) else 0.0


def reference_minerva_grade(solution: str, reference_answer: str) -> float:
    """The harness ``process_results`` exact_match metric, inlined."""
    candidate = reference_minerva.normalize_final_answer(reference_minerva.get_unnormalized_answer(solution))
    gold = reference_minerva.normalize_final_answer(reference_answer)
    return 1.0 if reference_minerva.is_equiv(candidate, gold) else 0.0


def test_hendrycks_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record["reference_answer"]
        for record in benchmark
        if hendrycks_math.grade(record["problem"], record["hendrycks_solution"], record["reference_answer"])
        != reference_hendrycks_grade(record["hendrycks_solution"], record["reference_answer"])
    ]
    assert mismatched == []


def test_minerva_matches_harness_on_benchmark(benchmark):
    for record in benchmark:
        got = minerva_math.grade(record["problem"], record["minerva_solution"], record["reference_answer"])
        want = reference_minerva_grade(record["minerva_solution"], record["reference_answer"])
        if got == want:
            continue
        # A comma list (bracketed or bare) has no faithful sympy reading: the
        # reference parses it as its first element or scores it 0 outright, while
        # the port compares the full list by normalized string. A disagreement is
        # only acceptable on such an answer.
        candidate = minerva_math.normalize_final_answer(
            minerva_math.extract_answer(record["minerva_solution"])
        )
        gold = minerva_math.normalize_final_answer(record["reference_answer"])
        assert minerva_math._LIST_COMMA.search(candidate) or minerva_math._LIST_COMMA.search(gold)


def test_hendrycks_normalization_matches_harness_on_benchmark_answers(benchmark):
    """``strip_string`` matches the harness except where a waiver applies."""
    for record in benchmark:
        answer = record["reference_answer"]
        got = hendrycks_math.strip_string(answer)
        want = reference_hendrycks.strip_string(answer)
        if got != want:
            # The port unwraps ``\\text{...}`` to its content; the reference
            # leaves the wrapper intact. That is the only normalization
            # divergence on this fixture.
            assert answer == "\\text{ellipse}"
            assert got == "ellipse"


def test_minerva_normalization_matches_harness_on_benchmark_answers(benchmark):
    for record in benchmark:
        answer = record["reference_answer"]
        got = minerva_math.normalize_final_answer(answer)
        want = reference_minerva.normalize_final_answer(answer)
        if got != want:
            # The port reduces ``\\dfrac`` to ``\\frac``; the reference keeps the
            # display variant. Those are the only normalization divergences on
            # this fixture.
            assert answer in {"\\dfrac{1}{12}", "\\dfrac{5}{162}"}
            assert got == want.replace("\\dfrac", "\\frac")
        extracted = minerva_math.extract_answer(record["minerva_solution"])
        assert extracted == reference_minerva.get_unnormalized_answer(record["minerva_solution"])


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        # A comma before three digits is a thousands separator, so this parses.
        ("(100,101)", "(100,101)"),
        ("(100,101)", "100101"),
        # The comma must be inside the brackets to prove a parse failure.
        ("(E),", "(E),"),
        # The grammar does have a comma production for call arguments, so a
        # bracketed function call parses and must not be rejected on sight.
        ("(f(x,y))", "(f(x,y))"),
        ("[f(x,y)]", "[f(x,y)]"),
        ("(2f(x,y))", "(2f(x,y))"),
        ("(x+f(a,b))", "(x+f(a,b))"),
        # Decimals compare exactly against rationals.
        ("0.5", "\\frac{1}{2}"),
        ("0.3", "\\frac{3}{10}"),
        ("0.333", "\\frac{1}{3}"),
        ("1.50", "1.5"),
        ("-0", "0"),
        ("2", "\\frac{4}{2}"),
        # Leading zeros are a Python int-literal syntax error inside sympy.
        ("007", "7"),
        ("\\frac{007}{2}", "\\frac{7}{2}"),
        # sympy's grammar wants a digit before the point.
        (".5", ".5"),
        # A parsed relation cannot be subtracted from itself.
        ("-80\\leqg(x)\\leq82", "-80\\leqg(x)\\leq82"),
    ],
)
def test_minerva_equivalence_matches_reference(candidate, reference):
    assert minerva_math.is_equiv(candidate, reference) is bool(reference_minerva.is_equiv(candidate, reference))


def waiver_cases():
    """The waiver patterns both graders must equate."""
    return [
        # Whitespace inside brackets.
        ("(1, 2)", "(1,2)"),
        ("( 1, 2 )", "(1,2)"),
        # Whitespace between operations.
        ("1 + 2i", "1+2i"),
        ("x + y + z", "x+y + z"),
        ("\\frac{a + b}{c}", "\\frac{a+b}{c}"),
        # A leading decimal point.
        (".12", "0.12"),
        # Redundant ``$`` as a math wrapper or a dollar unit.
        ("$2$", "2"),
        ("$$2$$", "2"),
        ("$2", "2"),
        # An extra text wrapper.
        ("\\text{Amy}", "Amy"),
        ("$Amy$", "Amy"),
        # LaTeX display-variant fractions.
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("\\tfrac{3}{4}", "\\frac{3}{4}"),
        # Multiple-choice letter formats.
        ("$E$", "E"),
        ("(E)", "E"),
        ("$(E)$", "E"),
        ("\\text{E}", "E"),
        ("\\text{(E)}", "E"),
        # Sizing bracket wrappers.
        ("\\left[ X, Y \\right]", "[X, Y]"),
        ("\\left(X, Y\\right)", "(X, Y)"),
        ("\\left( X,  Y \\right)", "(X, Y)"),
    ]


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        *waiver_cases(),
        # Fraction vs decimal; ``strip_string`` only special-cases the ``0.5``
        # form, the rest go through the sympy fallback.
        ("0.5", "\\frac{1}{2}"),
        # A bare base subscript braces to ``_{base}``, and a missing base is
        # not an error (``40`` == ``40_{9}``).
        ("40_9", "40"),
        ("40_{9}", "40"),
        # A missing unit is not an error (``15`` == ``15 \mbox{ cm^2}``),
        # whether the power sits inside or outside the unit wrapper.
        ("15", "15\\mbox{ cm^2}"),
        ("15", "15\\mbox{ cm}^2"),
        # Thousands-separator commas collapse via sympy.
        ("58500", "58,500"),
        ("11111111100", "11,111,111,100"),
        # General fraction vs decimal (not just the 0.5 special case).
        ("0.09", "\\frac{9}{100}"),
        ("5.5", "\\frac{11}{2}"),
        # Order of products.
        ("(b+2)(a+5)", "(a+5)(b+2)"),
    ],
)
def test_hendrycks_equivalences(candidate, reference):
    """Equivalent answers ``is_equiv`` accepts, via ``strip_string`` or sympy."""
    assert hendrycks_math.is_equiv(candidate, reference)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        *waiver_cases(),
        # A bracketed comma list is outside sympy's grammar, so the reference
        # scores even an identical string 0; the port now compares two such
        # lists by their normalized strings.
        ("[2,5)", "[2,5)"),
        # Commas that only group digits are separators.
        ("1,000,000", "1000000"),
        ("1,\\!000\\!000", "1000000"),
        # Fraction vs decimal.
        ("\\frac{1}{2}", "0.5"),
        ("\\frac{3}{10}", "0.3"),
        # Order of products.
        ("(a+b)(b+c)", "(b+c)(a+b)"),
        # A bare base subscript braces to ``_{base}``, and a missing base is
        # not an error (``40`` == ``40_{9}``).
        ("40_9", "40"),
        ("40_{9}", "40"),
        # A missing unit is not an error (``15`` == ``15 \mbox{ cm^2}``),
        # whether the power sits inside or outside the unit wrapper.
        ("15", "15\\mbox{ cm^2}"),
        ("15", "15\\mbox{ cm}^2"),
    ],
)
def test_minerva_equivalences(candidate, reference):
    """``grade``'s normalize-then-compare path equates each reformulation."""
    assert minerva_math.is_equiv(
        minerva_math.normalize_final_answer(candidate),
        minerva_math.normalize_final_answer(reference),
    )


def test_does_not_equate_different_bases():
    """Two different present bases are distinct: ``40_8`` is not ``40_9``.

    sympy's parser drops the subscript and would wrongly equate them; the string
    fast path keeps them apart.
    """
    assert not hendrycks_math.is_equiv("40_8", "40_9")
    assert not minerva_math.is_equiv(
        minerva_math.normalize_final_answer("40_8"),
        minerva_math.normalize_final_answer("40_9"),
    )


def test_minerva_keeps_full_equations_instead_of_collapsing_to_rhs():
    """A short binding splits to its RHS; a full equation is left whole."""
    assert minerva_math.normalize_final_answer("x = 5") == "5"
    assert minerva_math.normalize_final_answer("2x - 11y + 10z + 13 = 0") == "2x-11y+10z+13=0"
    assert minerva_math.normalize_final_answer("5x - 7y + 11z + 4 = 0") == "5x-7y+11z+4=0"


def test_minerva_does_not_equate_different_equations():
    """Two distinct ``... = 0`` equations are different planes, not both ``0``."""
    solution = "Final Answer: The final answer is $2x - 11y + 10z + 13 = 0$."
    assert minerva_math.grade("irrelevant problem text", solution, "5x - 7y + 11z + 4 = 0") == 0.0


def test_hendrycks_sympy_fallback_skips_comma_lists():
    """A comma list must not reach sympy, which parses it as its first element
    (``\frac{3}{4}, -\frac{3}{4}`` -> ``\frac{3}{4}``)."""
    assert not hendrycks_math.is_equiv("\\frac{3}{4}", "\\frac{3}{4}, -\\frac{3}{4}")
    assert not hendrycks_math.is_equiv("1, -2", "-2, 1")


def test_minerva_comma_list_skips_sympy():
    """A bare comma list must not reach sympy, which reads only its first element.

    ``-2`` must not equal ``-2,1`` (a list of two roots is not one root), and the
    guard holds for any number of elements: a list of one length never equals a
    list of another, while two identical lists do.
    """
    assert not minerva_math.is_equiv("-2", "-2,1")
    assert not minerva_math.is_equiv("-2", "-2,1,3")
    assert not minerva_math.is_equiv("-2,1", "-2,1,3")
    assert minerva_math.is_equiv("-2,1", "-2,1")
    assert minerva_math.is_equiv("-2,1,3", "-2,1,3")


def test_minerva_equates_identical_matrix_answers():
    """sympy cannot parse ``\\begin{...}``, so a matrix compares by string.

    The reference scores a matrix 0 even against itself; the port scores an
    identical matrix 1 while a different matrix or a scalar stays 0.
    """
    solution = "Final Answer: The final answer is $\\begin{pmatrix} -7 \\\\ 16 \\\\ 5 \\end{pmatrix}$."
    assert minerva_math.grade("irrelevant problem text", solution, "\\begin{pmatrix} -7 \\\\ 16 \\\\ 5 \\end{pmatrix}") == 1.0
    assert minerva_math.grade("irrelevant problem text", solution, "\\begin{pmatrix} -7 \\\\ 16 \\\\ 9 \\end{pmatrix}") == 0.0
    assert minerva_math.grade("irrelevant problem text", solution, "5") == 0.0


def test_hendrycks_strips_percent_escapes_created_by_an_earlier_removal():
    """``str.replace`` is single-pass, so the reference's repeated ``\\%`` strip matters.

    Collapsing ``\\\\`` to ``\\`` leaves ``\\\\%%``; one pass removes the middle
    ``\\%`` and leaves a newly adjacent one that only a second pass catches.
    """
    assert hendrycks_math.strip_string(r"\\\%%") == reference_hendrycks.strip_string(r"\\\%%")
    assert hendrycks_math.is_equiv(r"\\\%%", r"\\%") == bool(reference_hendrycks.is_equiv(r"\\\%%", r"\\%"))


def test_hendrycks_falls_back_to_raw_equality_when_normalization_raises():
    """A non-integer ``a/b`` makes ``int()`` raise, dropping ``is_equiv`` to raw equality."""
    assert hendrycks_math.is_equiv("x/y", "x/y")
    assert not hendrycks_math.is_equiv("x/y", "x/y ")


def test_graders_need_only_problem_solution_and_reference():
    """No dataset doc, no harness state."""
    assert (
        minerva_math.grade(
            "irrelevant problem text",
            "Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.",
            "0.5",
        )
        == 1.0
    )
    assert hendrycks_math.grade("irrelevant problem text", " $0.5$", "\\frac{1}{2}") == 1.0
