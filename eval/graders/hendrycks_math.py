"""Self-contained grader for the lm-eval-harness ``hendrycks_math`` task.

Ports ``lm_eval/tasks/hendrycks_math/utils.py`` at the pinned
lm-evaluation-harness ``v0.4.12``: grading needs only the problem text, the
model's solution, and the reference answer. ``eval/chat_benchmarks/MATH500``
imports ``is_equiv`` from the harness module directly; this port is a drop-in
replacement for it.

Equivalence is pure string normalization -- two answers match when
``strip_string`` maps them to the same text -- and already costs ~2 us per
problem, so the chain is reproduced verbatim rather than tuned, with three
string-level additions: a bare ``$`` is stripped (a redundant math-mode
delimiter or a dollar unit), a ``\\text{...}`` wrapper is unwrapped to its
content, and a parenthesized single letter ``(E)`` is unwrapped to ``E`` (a
multiple-choice answer format). That includes the points where the chain raises:
``is_equiv`` swallows those and falls back to raw string equality, making the
raising behavior part of the graded contract.

When the two normalized strings differ, ``is_equiv`` defers to a sympy
``simplify(candidate - reference) == 0`` check (the same one ``minerva_math``
uses) so the patterns a string cannot express -- reordered products, a fraction
against its decimal, thousands-separator commas -- still compare equal. The
string fast path is always tried first; sympy runs only on the rarer mismatches.
"""

import functools
import re
import signal

import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

_SYMPY_TIMEOUT = 5
_PARSE_CACHE_SIZE = 8192

_TEXT = re.compile(r"\\text\{([^}]*)\}")
# A base-subscript answer ``X_b`` (e.g. ``40_9``) braces to ``X_{b}``; the base
# number itself is digits optionally followed by a braced base ``_{b}``.
_BRACE_BASE_SUBSCRIPT = re.compile(r"^(\d+)_(\d+)$")
_BASE_NUMBER = re.compile(r"^(\d+)(?:_\{(\d+)\})?$")
# A comma that is not a thousands separator (not followed by three digits)
# marks a list/tuple answer; sympy parses such a string as its first element, so
# the sympy fallback must not run on it.
_LIST_COMMA = re.compile(r",(?!\d{3})")


def extract_answer(solution: str) -> str:
    """Return the span between the first and last ``$`` in a completion.

    Mirrors the extraction in ``process_results``: with fewer than two ``$``
    the whole completion is the answer.
    """
    first = solution.find("$")
    last = solution.rfind("$")
    if first == last:
        return solution
    return solution[first + 1 : last]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            post_substr = substr[2:]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + post_substr
            else:
                new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string: str) -> str:
    """Rewrite an integer ``a/b`` as ``\\frac{a}{b}``.

    Only ``AssertionError`` is absorbed, matching the reference: a non-integer
    operand lets ``ValueError`` escape and drops ``is_equiv`` to raw equality.
    """
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a = int(parts[0])
    b = int(parts[1])
    try:
        assert string == f"{a}/{b}"
    except AssertionError:
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def remove_right_units(string: str) -> str:
    """Drop a trailing ``\\text{ ...}`` unit.

    Raises ``AssertionError`` on a second ``\\text{ ``, which drops ``is_equiv``
    to raw equality.
    """
    if "\\text{ " not in string:
        return string
    splits = string.split("\\text{ ")
    assert len(splits) == 2
    return splits[0]


def fix_sqrt(string: str) -> str:
    """Brace a bare ``\\sqrt`` argument: ``\\sqrt3`` becomes ``\\sqrt{3}``.

    A trailing ``\\sqrt`` with no argument raises ``IndexError``, which drops
    ``is_equiv`` to raw equality.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    # The reference's string chain already covers most Part 5 / Part 6 waivers;
    # the additions below (bare ``$`` strip, ``\\text{...}`` unwrap, ``(E)``
    # unwrap) close the rest. See ``task_output/task_grader_implementation.overview.md``
    # for the full map.
    string = string.replace("\n", "")
    # Part 5 #3: drop inverse spaces (``\\!``) so a ``\\!``-separated number
    # like ``1,\\!000\\!000`` collapses toward ``1,000,000``.
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    # Part 6 #10: display-variant fractions reduce to ``frac``.
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # Part 6 #12: strip sizing wrappers so ``\\left(X, Y\\right)`` -> ``(X, Y)``.
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    # Part 6 #7/#8: strip a bare ``$`` (redundant math delimiter or dollar
    # unit), so ``$2$``, ``$$2$$`` and ``$2`` all -> ``2``.
    string = string.replace("$", "")
    # Part 11 #12: ``\mbox{...}`` is a unit wrapper like ``\text{...}``; fold it
    # so remove_right_units strips both (``15`` == ``15\mbox{ cm}^2``).
    string = string.replace("\\mbox", "\\text")
    string = remove_right_units(string)
    # Part 6 #9: unwrap a text wrapper to its content, ``\\text{E}`` -> ``E``.
    string = _TEXT.sub(r"\1", string)
    # The reference strips "\%" twice and never strips a bare "%" (its second
    # literal is an invalid escape Python leaves as backslash-percent). The
    # repeat is not redundant: str.replace is single-pass, so removing one match
    # can leave a newly adjacent one that only a second pass catches.
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    # Part 5 #4: give a leading decimal point its zero, ``.12`` -> ``0.12``.
    if string[0] == ".":
        string = "0" + string

    # Drop a short variable binding such as "k =" or "q =".
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)
    # Part 5 #1/#2: remove spaces so bracket/operation whitespace collapses,
    # e.g. ``(1, 2)`` -> ``(1,2)`` and ``1 + 2i`` -> ``1+2i``.
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    # Part 6 #11: unwrap a parenthesized multiple-choice letter, ``(E)`` -> ``E``.
    if (
        len(string) == 3
        and string[0] == "("
        and string[2] == ")"
        and "A" <= string[1] <= "Z"
    ):
        string = string[1]
    # Part 11 #11: brace a bare base subscript, ``4210_5`` -> ``4210_{5}``.
    string = _BRACE_BASE_SUBSCRIPT.sub(r"\1_{\2}", string)
    return fix_a_slash_b(string)


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


@functools.lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _sympy_parses(string: str):
    """Parse a LaTeX string, or return ``None`` if sympy rejects it.

    Memoized: parsing is pure and answers repeat across a run.
    """
    try:
        return parse_latex(string)
    except (LaTeXParsingError, sympy.SympifyError, TypeError):
        return None


def _sympy_is_equiv(candidate: str, reference: str) -> bool:
    """Parse both sides and simplify the difference, as ``minerva_math`` does.

    This is the fallback for Part 5 #3/#5/#6 (thousands separators,
    fraction-vs-decimal, product order) that a string comparison cannot express.
    Identical strings are already short-circuited by the caller, so a parsed
    relation such as ``a \\leq b`` still scores 0 here.
    """

    def on_timeout(signum, frame):
        raise TimeoutError

    previous = None
    try:
        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(_SYMPY_TIMEOUT)
        parsed_candidate = _sympy_parses(candidate)
        if parsed_candidate is None:
            return False
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


def is_equiv(candidate: str | None, reference: str | None) -> bool:
    if candidate is None and reference is None:
        return True
    if candidate is None or reference is None:
        return False
    try:
        stripped_candidate = strip_string(candidate)
        stripped_reference = strip_string(reference)
        if stripped_candidate == stripped_reference:
            return True
        # Part 11 #11: a missing base compares equal, different bases do not.
        # When both sides are base numbers, decide here and do not fall through
        # to sympy, whose parser drops the subscript and would equate ``40_8``
        # with ``40_9``.
        if _BASE_NUMBER.match(stripped_candidate) and _BASE_NUMBER.match(stripped_reference):
            return _base_numbers_equal(stripped_candidate, stripped_reference)
        # A list/tuple answer must not reach sympy: it would parse as its first
        # element (``\frac{3}{4}, -\frac{3}{4}`` -> ``\frac{3}{4}``).
        if _LIST_COMMA.search(stripped_candidate) or _LIST_COMMA.search(stripped_reference):
            return False
        # Part 5 #3/#5/#6: the string comparison cannot reorder products,
        # reduce a fraction to its decimal, or drop thousands-separator commas,
        # so defer to sympy's simplify-of-the-difference.
        return _sympy_is_equiv(stripped_candidate, stripped_reference)
    except Exception:
        return candidate == reference


def grade(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion against the reference answer.

    Args:
        problem: Problem statement. Unused; part of the shared grader signature.
        solution: The model's completion.
        reference_answer: Gold answer, already unwrapped from ``\\boxed{...}``.

    Returns:
        ``1.0`` when the completion matches the reference, else ``0.0``.
    """
    return 1.0 if is_equiv(extract_answer(solution), reference_answer) else 0.0
