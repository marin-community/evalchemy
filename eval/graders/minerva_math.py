"""Self-contained grader for the lm-eval-harness ``minerva_math`` task.

Ports the ``exact_match`` metric of ``lm_eval/tasks/minerva_math/utils.py`` at
the pinned lm-evaluation-harness ``v0.4.12`` so grading needs only the problem
text, the model's solution text, and the reference answer.

The reference parses both sides with sympy's ANTLR LaTeX parser and simplifies
the difference; ~98% of its runtime is ``parse_latex`` (2.4 ms per call, twice
per grade) against ~0.05 ms for ``simplify``. This module returns the same
verdicts while skipping the parser wherever the answer shape already settles
the question, and falls through to sympy otherwise. The shape rules are
documented at each pattern below.
"""

import functools
import re
import signal
from fractions import Fraction

import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

INVALID_ANSWER = "[invalidanswer]"
_SYMPY_TIMEOUT = 5
_PARSE_CACHE_SIZE = 8192

_FINAL_ANSWER = re.compile(
    r"Final Answer: The final answer is(.*?)(?:\. I hope it is correct\.|\.)\s*$"
)

SUBSTITUTIONS = [
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    r"\left",
    r"\right",
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    # Part 11 #12: a power written *inside* the unit wrapper (``\text{cm^2}``)
    # leaves ``\text{^2}`` once the ``cm`` word above is removed; drop it so the
    # bare value remains (``15`` == ``15\text{ cm^2}``). The ``\text{}^2``
    # entries above cover the power-outside form ``\text{cm}^2``.
    "\\text{^2}",
    "\\text{^3}",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    r"\!",
    "{,}",
    '"',
    "\\dots",
]

_TEXT = re.compile(r"(\\text\{)(.*?)(\})")
_TEXTBF = re.compile(r"(\\textbf\{)(.*?)(\})")
_OVERLINE = re.compile(r"(\\overline\{)(.*?)(\})")
_BOXED = re.compile(r"(\\boxed\{)(.*)(\})")
_SHORT_FRAC = re.compile(r"(frac)([^{])(.)")
_SHORT_SQRT = re.compile(r"(sqrt)([^{])")

# Bracketed comma lists (tuples, intervals) have no production in sympy's LaTeX
# grammar, so the reference scores them 0 against anything. Four details make
# the pattern sound: a bare "1,2,3" parses as its first element rather than
# failing; a comma followed by three digits is a thousands separator, so
# "(100,101)" lexes as 100101 and parses; the comma must fall inside the
# brackets, since trailing punctuation like "(E)," parses fine; and the grammar
# does carry a comma production for function-call arguments
# (args: (expr ',' args) | expr), so "(f(x,y))" parses. Barring any inner
# bracket from the run before the comma leaves those to sympy.
_BRACKETED_TUPLE = re.compile(r"^(?:\\left)?[(\[][^()\[\]]*,(?!\d{3})")

# A comma that is not a thousands separator (not followed by three digits) marks
# a bare list answer (``-2,1``). sympy parses such a string as its first element
# and drops the tail, so the fallback must not run on it. ``_BRACKETED_TUPLE``
# above covers the bracketed form; this catches the unbracketed one.
_LIST_COMMA = re.compile(r",(?!\d{3})")

# A ``\begin{...}`` environment (matrix, cases, aligned, ...) has no production
# in sympy's LaTeX grammar, so the reference scores it 0 even against an
# identical string. Part 9: compare two such answers by their normalized strings
# instead, so an identical matrix (e.g. a boxed ``\begin{pmatrix}``) scores 1.
_BEGIN_ENV = re.compile(r"\\begin\{")

# Integers reach sympy as Python int literals, so a leading zero is a
# SyntaxError: "007" and "\frac{007}{2}" both fail to parse. Decimals skip that
# path ("00.5" is fine) but need a digit before the point (".2" fails). A
# leading "+" is left to the fallback rather than assumed parseable.
_UNSIGNED_INT = r"(?:0|[1-9][0-9]*)"
_INTEGER = re.compile(rf"^-?{_UNSIGNED_INT}$")
_DECIMAL = re.compile(r"^-?[0-9]+\.[0-9]+$")
_FRACTION = re.compile(rf"^(?P<sign>-?)\\frac\{{(?P<num>-?{_UNSIGNED_INT})\}}\{{(?P<den>-?{_UNSIGNED_INT})\}}$")

# A base-subscript answer ``X_b`` (e.g. ``40_9``) braces to ``X_{b}``; the base
# number itself is digits optionally followed by a braced base ``_{b}``.
_BRACE_BASE_SUBSCRIPT = re.compile(r"^(\d+)_(\d+)$")
_BASE_NUMBER = re.compile(r"^(\d+)(?:_\{(\d+)\})?$")

# simplify() rationalizes a Float against a Rational at sympy's default 15-digit
# precision, so decimals compare exactly as Fractions only up to that width.
_FLOAT_SIGNIFICANT_DIGITS = 15


def extract_answer(solution: str) -> str:
    """Pull the answer out of a minerva-style completion.

    Returns ``INVALID_ANSWER`` when the completion does not contain the
    ``Final Answer: The final answer is ...`` template. Both the harness
    format (``... . I hope it is correct.``) and the reformatted fixture
    format (``... .``) are accepted.
    """
    match = _FINAL_ANSWER.search(solution)
    return match.group(1).strip() if match else INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """Normalize an answer per appendix D of Lewkowycz et al. (2022).

    Departures from the reference fix matching bugs the reference leaves open,
    and add the Part 5 / Part 6 equivalence waivers (the full map lives in
    ``task_output/task_grader_implementation.overview.md``):

    * leading article stripped only at the start -- ``a 3`` -> ``3`` but
      ``a + b`` keeps its variable ``a``;
    * leading decimal point gains a zero -- ``.12`` -> ``0.12`` (Part 5 #4);
    * ``$`` stripped uniformly -- ``$2$``, ``$$2$$``, ``$2`` -> ``2``
      (Part 6 #7, #8);
    * ``\\dfrac``/``\\tfrac`` -> ``\\frac`` -- display variants (Part 6 #10);
    * ``(E)`` -> ``E`` -- a multiple-choice letter (Part 6 #11);
    * the right-hand side of a binding is kept only when the left side is a
      short variable name -- ``x = 5`` -> ``5`` but an equation
      ``Ax + By + Cz + D = 0`` stays whole (the reference collapses every
      such answer to ``0`` and equates different equations).

    Part 5 #1-#3, #5, #6 and Part 6 #9, #12 are handled here and in
    ``is_equiv``; see the overview doc for where each lives.
    """
    # A short LHS ("x = 5") is a binding; keep only its RHS. A longer LHS is a
    # full equation ("Ax + By + Cz + D = 0"), which the reference collapses to
    # "0" — keep it whole so two different equations stay distinct. Stripping
    # "$" and whitespace makes the check symmetric across "$y = -2x$" and
    # "y = -2x".
    parts = final_answer.split("=")
    if len(parts) == 2 and len(parts[0].replace("$", "").strip()) <= 2:
        final_answer = parts[1]

    # Strip a leading article only, not every "a " as the reference did (which
    # mangled ``a + b`` into ``+b``).
    if final_answer.startswith("an "):
        final_answer = final_answer[3:]
    elif final_answer.startswith("a "):
        final_answer = final_answer[2:]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Part 6 #9: unwrap a text/format wrapper to its content, e.g.
    # ``\\text{E}`` -> ``E``. (``\\left``/``\\right`` for Part 6 #12 are removed
    # above via REMOVED_EXPRESSIONS.)
    final_answer = _TEXT.sub("\\2", final_answer)
    final_answer = _TEXTBF.sub("\\2", final_answer)
    final_answer = _OVERLINE.sub("\\2", final_answer)
    final_answer = _BOXED.sub("\\2", final_answer)

    final_answer = _SHORT_FRAC.sub("frac{\\2}{\\3}", final_answer)
    final_answer = _SHORT_SQRT.sub("sqrt{\\2}", final_answer)
    # Part 6 #7/#8: strip every ``$`` uniformly (redundant math delimiter or
    # dollar unit), so ``$2$``, ``$$2$$`` and ``$2`` all -> ``2``.
    final_answer = final_answer.replace("$", "")

    # Part 5 #4: give a leading decimal point its zero, e.g. ``.12`` -> ``0.12``.
    if final_answer.startswith(".") and final_answer[1:2].isdigit():
        final_answer = "0" + final_answer
    elif final_answer.startswith("-.") and final_answer[2:3].isdigit():
        final_answer = "-0." + final_answer[2:]

    # Part 5 #3: commas that only group digits are separators, e.g.
    # ``1,000,000`` -> ``1000000``.
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    # Part 6 #11: unwrap a parenthesized multiple-choice letter, ``(E)`` -> ``E``.
    if (
        len(final_answer) == 3
        and final_answer[0] == "("
        and final_answer[2] == ")"
        and "A" <= final_answer[1] <= "Z"
    ):
        final_answer = final_answer[1]
    # Part 11 #11: brace a bare base subscript, ``40_9`` -> ``40_{9}``.
    final_answer = _BRACE_BASE_SUBSCRIPT.sub(r"\1_{\2}", final_answer)
    return final_answer


def _rational_value(string: str) -> Fraction | None:
    """Exact value of a plain integer, decimal, or integer fraction, else ``None``."""
    if _INTEGER.match(string):
        return Fraction(int(string))

    if _DECIMAL.match(string):
        digits = string.lstrip("+-").replace(".", "").lstrip("0")
        if len(digits) > _FLOAT_SIGNIFICANT_DIGITS:
            return None
        return Fraction(string)

    match = _FRACTION.match(string)
    if match is None:
        return None
    denominator = int(match["den"])
    if denominator == 0:
        return None
    value = Fraction(int(match["num"]), denominator)
    return -value if match["sign"] else value


@functools.lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _sympy_parses(string: str):
    """Parse a LaTeX string, or return ``None`` if sympy rejects it.

    Memoized: parsing is pure and answers repeat across a run (MATH's 12.5k
    answers cover 4.3k distinct strings).
    """
    try:
        return parse_latex(string)
    except (LaTeXParsingError, sympy.SympifyError, TypeError):
        return None


def _sympy_is_equiv(candidate: str, reference: str) -> bool:
    """Parse both sides and simplify the difference, as the reference does.

    Identical strings need only one parse, but still take the difference: a
    parsed relation such as ``a\\leq b`` cannot be subtracted from itself and so
    scores 0.
    """

    def on_timeout(signum, frame):
        raise TimeoutError

    # Arming the alarm sits inside the guard: off the main thread signal.signal
    # raises, which the reference also scores as a non-match.
    previous = None
    try:
        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(_SYMPY_TIMEOUT)
        parsed_candidate = _sympy_parses(candidate)
        if parsed_candidate is None:
            return False
        if candidate == reference:
            parsed_reference = parsed_candidate
        else:
            parsed_reference = _sympy_parses(reference)
            if parsed_reference is None:
                return False
        return sympy.simplify(parsed_candidate - parsed_reference) == 0
    except ImportError:
        # Without antlr4-python3-runtime 4.11 every comparison would quietly
        # score 0, so surface it rather than report a non-match.
        raise
    except Exception:
        # Parse, subtraction, simplify, and timeout failures all score 0.
        return False
    finally:
        if previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _base_numbers_equal(candidate: str, reference: str) -> bool:
    """Compare two normalized base numbers, ignoring a missing base.

    A base on only one side compares equal (``40`` == ``40_{9}``); two different
    present bases do not (``40_8`` != ``40_9``). Returns ``False`` when either
    side is not a base number, so it never overrides an expression match.
    """
    candidate_match = _BASE_NUMBER.match(candidate)
    reference_match = _BASE_NUMBER.match(reference)
    if candidate_match is None or reference_match is None:
        return False
    if candidate_match.group(1) != reference_match.group(1):
        return False
    candidate_base = candidate_match.group(2)
    reference_base = reference_match.group(2)
    return candidate_base is None or reference_base is None or candidate_base == reference_base


def is_equiv(candidate: str, reference: str) -> bool:
    """Decide equivalence of two already-normalized LaTeX answers.

    Fast paths cover Part 5 patterns without a sympy parse:

    * #1/#12 -- bracketed comma lists compare by normalized string
      (``(1, 2)`` == ``(1,2)``);
    * bare comma lists (``-2,1``) compare by normalized string too -- sympy
      parses such a string as its first element, so a list must not fall through
      to the parser (``-2`` is not ``-2,1``, and a reordered list is distinct);
    * #5 -- integers, decimals and integer ``\\frac`` forms compare as exact
      ``Fraction`` values (``\\frac{1}{2}`` == ``0.5``);
    * ``\\begin{...}`` environments (matrices) compare by normalized string
      (``\\begin{pmatrix}-7\\\\16\\\\5\\end{pmatrix}`` == itself);
    * Part 11 #11 -- base numbers (``40``, ``40_{9}``) compare by digits plus
      optional base, so a missing base matches while two different bases do not.
    * a full equation (``=`` with a long LHS, kept whole by
      ``normalize_final_answer``) compares by normalized string -- sympy cannot
      subtract one ``Eq`` relation from another, so it would score even an
      identical equation 0.

    Anything else falls through to sympy, where ``simplify`` of the difference
    covers #2 whitespace-between-operations and #6 product order.
    """
    candidate_is_tuple = _BRACKETED_TUPLE.match(candidate) is not None
    reference_is_tuple = _BRACKETED_TUPLE.match(reference) is not None
    if candidate_is_tuple or reference_is_tuple:
        # Part 5 #1 (and #12, whose \\left/\\right and space stripping already
        # ran in normalize_final_answer): tuples/intervals have no sympy
        # production, so compare them directly. Exact string equality catches
        # ``(1, 2)`` vs ``(1,2)`` while a lone tuple still scores 0.
        return candidate_is_tuple and reference_is_tuple and candidate == reference

    # A bare comma list (``-2,1``) has no top-level sympy production: parse_latex
    # reads only its first element and drops the tail, so ``-2`` would equal
    # ``-2,1``. Compare two such lists by string instead -- identical only, so a
    # list against a scalar or a reordered list scores 0. (The bracketed form is
    # handled above by _BRACKETED_TUPLE.)
    candidate_is_list = _LIST_COMMA.search(candidate) is not None
    reference_is_list = _LIST_COMMA.search(reference) is not None
    if candidate_is_list or reference_is_list:
        return candidate_is_list and reference_is_list and candidate == reference

    # ``\begin{...}`` environments have no sympy production, so compare them
    # directly: a lone one scores 0, two identical ones score 1.
    candidate_is_env = _BEGIN_ENV.search(candidate) is not None
    reference_is_env = _BEGIN_ENV.search(reference) is not None
    if candidate_is_env or reference_is_env:
        return candidate_is_env and reference_is_env and candidate == reference

    # A full equation kept whole by normalize_final_answer (``... = 0``) has no
    # relation arithmetic in sympy: subtracting one ``Eq`` from another fails, so
    # the fallback scores even an identical equation 0. Compare by string instead.
    candidate_is_equation = "=" in candidate
    reference_is_equation = "=" in reference
    if candidate_is_equation or reference_is_equation:
        return candidate_is_equation and reference_is_equation and candidate == reference

    # Part 11 #11: base numbers compare by string fast path. A missing base
    # (``40`` vs ``40_{9}``) matches while two different present bases
    # (``40_8`` vs ``40_9``) do not; sympy drops the subscript and would wrongly
    # equate the latter.
    if _BASE_NUMBER.match(candidate) and _BASE_NUMBER.match(reference):
        return _base_numbers_equal(candidate, reference)

    # Part 5 #5: exact rational comparison skips sympy for plain numbers.
    candidate_value = _rational_value(candidate)
    if candidate_value is not None:
        reference_value = _rational_value(reference)
        if reference_value is not None:
            return candidate_value == reference_value

    # Part 5 #2 and #6: sympy parses both sides and simplifies the difference,
    # erasing operation whitespace (``1 + 2i`` == ``1+2i``) and product order
    # (``(a+b)(b+c)`` == ``(b+c)(a+b)``).
    return _sympy_is_equiv(candidate, reference)


def grade(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion against the reference answer.

    Args:
        problem: Problem statement. Unused; part of the shared grader signature.
        solution: The model's completion.
        reference_answer: Gold answer, already unwrapped from ``\\boxed{...}``.

    Returns:
        ``1.0`` when the completion matches the reference, else ``0.0``.
    """
    candidate = normalize_final_answer(extract_answer(solution))
    reference = normalize_final_answer(reference_answer)
    return 1.0 if is_equiv(candidate, reference) else 0.0
