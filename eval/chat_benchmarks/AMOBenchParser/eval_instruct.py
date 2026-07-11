"""Parser-gradeable subset of AMO-Bench."""

from typing import Any, Dict, List, Tuple

from eval.chat_benchmarks.AMOBenchParser.solver import solve_many_with_timeout
from eval.chat_benchmarks.final_answer_math import FinalAnswerMathBenchmark, GradeOutcome


ANSWER_PREFIXES = [
    "### the final answer is:",
    "### the final answer:",
    "### final answer is:",
    "### final answer:",
    "### the final answer is",
    "### the final answer",
    "### final answer is",
    "### final answer",
]
THINK_POSTFIXES = ["</think>", "</longcat_think>"]
CUT_MARKERS = ["\\medskip", "\n---"]
REMOVE_TOKENS = [
    "\\bigl",
    "\\bigr",
    "\\Bigl",
    "\\Bigr",
    "\\biggl",
    "\\biggr",
    "\\Biggl",
    "\\Biggr",
    "\\bigg",
    "\\Bigg",
    "\\big",
    "\\Big",
    "\\left",
    "\\right",
]
REPLACEMENTS = [
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\uff08", "("),
    ("\uff09", ")"),
    ("\uff0c", ", "),
    ("\uff1a", ": "),
    ("\uff1b", "; "),
    ("\u3002", ". "),
    ("\uff01", "! "),
    ("\uff1f", "? "),
    ("\u2026", "..."),
    ("\u2013", "-"),
    ("\u2212", "-"),
]


def extract_amo_prediction(prediction: str, answer_type: str) -> str:
    extracted = prediction.replace("\uff1a", ": ")
    for postfix in THINK_POSTFIXES:
        extracted = extracted.split(postfix)[-1].strip()

    prefixes = ANSWER_PREFIXES + [prefix[4:] for prefix in ANSWER_PREFIXES]
    for prefix in prefixes:
        if prefix in extracted.lower():
            lower_tail = extracted.lower().split(prefix)[-1]
            extracted = extracted[-len(lower_tail) :].strip()
            break

    if answer_type != "description":
        for token in REMOVE_TOKENS:
            extracted = extracted.replace(token, "")
    for old, new in REPLACEMENTS:
        extracted = extracted.replace(old, new)

    while " }" in extracted:
        extracted = extracted.replace(" }", "}")
    while ".}" in extracted:
        extracted = extracted.replace(".}", "}")

    if answer_type in {"number", "variable", "set"}:
        extracted = extracted.replace("\\,", "").replace("\\;", "").replace("\n", " ")
    if answer_type in {"number", "variable"}:
        extracted = extracted.replace(",", "")
        extracted = extracted.replace("\\{", "(").replace("\\}", ")")
        extracted = extracted.replace("\\[", "(").replace("\\]", ")")
    return extracted.strip()


def cut_prediction(prediction: str) -> str:
    for marker in CUT_MARKERS:
        prediction = prediction.split(marker)[0].strip()
    return prediction


def _require_grader_dependencies():
    try:
        from math_verify import parse, verify
        from sympy import solve
    except ImportError as exc:
        raise RuntimeError(
            "AMOBenchParser requires the math extra. Install it with `pip install -e '.[math]'`."
        ) from exc
    return parse, verify, solve


def _verify_number_or_set(prediction: str, example: Dict[str, Any]) -> bool:
    parse, verify, _ = _require_grader_dependencies()
    pred_parse = parse(prediction, parsing_timeout=None)
    gold_parse = parse(example["answer"], parsing_timeout=None)
    result = bool(
        verify(gold_parse, pred_parse, float_rounding=4, timeout_seconds=None)
        or verify(pred_parse, gold_parse, float_rounding=4, timeout_seconds=None)
    )

    if pred_parse and "=" in str(pred_parse[-1]):
        last_value = str(pred_parse[-1]).split("=")[-1]
        last_parse = parse("\\boxed{" + last_value + "}", parsing_timeout=None)
        result = result or bool(
            verify(gold_parse, last_parse, float_rounding=4, timeout_seconds=None)
            or verify(last_parse, gold_parse, float_rounding=4, timeout_seconds=None)
        )
    return result


def _solution_for_symbol(solution: Any, symbol_name: str):
    if isinstance(solution, list):
        if not solution:
            return None
        solution = solution[0]
    if not hasattr(solution, "items"):
        return None
    for symbol, value in solution.items():
        if str(symbol) == symbol_name:
            return value
    return None


def _verify_variable(prediction: str, example: Dict[str, Any]) -> bool:
    parse, verify, solve = _require_grader_dependencies()
    pred_original = parse(prediction, parsing_timeout=None)
    if not pred_original:
        return False

    pred_expression = str(pred_original[-1])
    pred_expression = pred_expression.split("\\qquad")[-2].strip() if "\\qquad" in pred_expression else pred_expression
    pred_expression = pred_expression.split("\\quad")[-2].strip() if "\\quad" in pred_expression else pred_expression
    pred_expression = pred_expression.split("=")[-1]

    gold_original = parse(example["answer"], parsing_timeout=None)
    gold_expression = str(gold_original[-1]).split("=")[-1]
    pred_equations = []
    gold_solutions = []
    for test_case in example["verification_cases"]:
        pred_equation = parse(
            "\\boxed{" + test_case + ", y=" + pred_expression + "}",
            parsing_timeout=None,
        )
        gold_equation = parse(
            "\\boxed{" + test_case + ", y=" + gold_expression + "}",
            parsing_timeout=None,
        )
        pred_equations.append(pred_equation[0])
        gold_solutions.append(solve(gold_equation[0]))

    pred_solutions = solve_many_with_timeout(pred_equations)
    for pred_solution, gold_solution in zip(pred_solutions, gold_solutions):
        pred_y = _solution_for_symbol(pred_solution, "y")
        gold_y = _solution_for_symbol(gold_solution, "y")
        if pred_y is None or gold_y is None:
            return False
        if not (
            verify(gold_y.evalf(), pred_y.evalf(), float_rounding=8, timeout_seconds=None)
            or verify(pred_y.evalf(), gold_y.evalf(), float_rounding=8, timeout_seconds=None)
        ):
            return False
    return True


def verify_amo_prediction(prediction: str, example: Dict[str, Any]) -> bool:
    if example["answer_type"] in {"number", "set"}:
        return _verify_number_or_set(prediction, example)
    if example["answer_type"] == "variable":
        return _verify_variable(prediction, example)
    raise ValueError(f"Unsupported parser answer type: {example['answer_type']}")


class AMOBenchParserBenchmark(FinalAnswerMathBenchmark):
    DEFAULT_DATA_FILE = "eval/chat_benchmarks/AMOBenchParser/data/amo_bench_parser.jsonl"
    EXPECTED_ROWS = 39
    REQUIRED_FIELDS = ("question_id", "problem", "answer", "answer_type")
    ID_FIELD = "question_id"
    GROUP_FIELD = "answer_type"
    USE_SOURCE_PROMPT = True

    def extract_answer(self, output: str, example: Dict[str, Any] = None) -> str:
        return extract_amo_prediction(output, example["answer_type"])

    def grade_answer(self, gold: str, prediction: str, example: Dict[str, Any]) -> GradeOutcome:
        if not prediction:
            return GradeOutcome(False, "missing_answer")

        _require_grader_dependencies()
        attempts: List[Tuple[str, str]] = [("amo_parser", prediction)]
        cut = cut_prediction(prediction)
        if cut != prediction:
            attempts.append(("amo_parser_cut", cut))
        for method, candidate in attempts:
            try:
                if verify_amo_prediction(candidate, example):
                    return GradeOutcome(True, method)
            except Exception:
                continue
        return GradeOutcome(False, "amo_parser")
