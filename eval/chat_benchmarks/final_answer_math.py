"""Shared generation and scoring scaffold for final-answer math benchmarks."""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


DEFAULT_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n" "Problem: {problem}"
OLYMMATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}"


@dataclass(frozen=True)
class GradeOutcome:
    correct: bool
    method: str


def extract_last_boxed(text: str) -> str:
    """Reproduce OlymMATH's final balanced ``\\boxed{...}`` extraction."""
    stack = []
    answers: List[str] = []
    idx = 0
    answer_start = -1

    while idx < len(text):
        if text[idx : idx + 7] == "\\boxed{" and (idx == 0 or text[idx - 1] != "\\"):
            if not stack:
                answer_start = idx + 7
            stack.append("{")
            idx += 7
        elif text[idx] == "{" and (idx == 0 or text[idx - 1] != "\\"):
            stack.append("{")
            idx += 1
        elif text[idx] == "}" and (idx == 0 or text[idx - 1] != "\\"):
            if stack:
                stack.pop()
                if not stack and answer_start != -1:
                    answers.append(text[answer_start:idx])
                    answer_start = -1
            idx += 1
        else:
            idx += 1

    if answers:
        return answers[-1]

    # This fallback is part of the released OlymMATH extractor and recovers some
    # malformed or double-escaped model outputs that the stack pass cannot close.
    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*?)}"
    matches = list(re.finditer(pattern, text))
    return matches[-1].group(1) if matches else ""


def strip_outer_math_delimiters(answer: str) -> str:
    answer = str(answer).strip()
    pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    for opening, closing in pairs:
        if answer.startswith(opening) and answer.endswith(closing) and len(answer) >= len(opening) + len(closing):
            return answer[len(opening) : -len(closing)].strip()
    return answer


def exact_string_grade(gold: str, prediction: str) -> GradeOutcome:
    if not prediction:
        return GradeOutcome(False, "missing_answer")
    return GradeOutcome(
        strip_outer_math_delimiters(prediction) == strip_outer_math_delimiters(gold),
        "exact_string",
    )


def _require_math_verify():
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires the math extra. Install it with `pip install -e '.[math]'`."
        ) from exc
    return parse, verify


def _format_for_math_verify(answer: str) -> str:
    if not answer:
        return "$.$"
    answer = str(answer).strip()
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]
    answer = answer.strip()
    return f"${answer}$" if answer else "$.$"


def _olymmath_normalize(answer: str) -> str:
    normalized = re.sub(r"\s+", "", answer or "")
    normalized = normalized.replace("\\frac", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    return re.sub(r"\\[a-zA-Z]+", "", normalized)


def olymmath_string_fallback(gold: str, prediction: str) -> bool:
    """Reproduce the normalized equality/inclusion fallback in OlymMATH."""
    normalized_gold = _olymmath_normalize(gold)
    normalized_prediction = _olymmath_normalize(prediction)
    return (
        normalized_prediction == normalized_gold
        or normalized_gold in normalized_prediction
        or normalized_prediction in normalized_gold
    )


def olymmath_grade(gold: str, prediction: str) -> GradeOutcome:
    if not prediction:
        return GradeOutcome(False, "missing_answer")

    parse, verify = _require_math_verify()
    try:
        gold_parsed = parse(_format_for_math_verify(gold), parsing_timeout=None)
        prediction_parsed = parse(_format_for_math_verify(prediction), parsing_timeout=None)
        return GradeOutcome(
            bool(verify(gold_parsed, prediction_parsed, timeout_seconds=None)),
            "math_verify",
        )
    except Exception:
        return GradeOutcome(olymmath_string_fallback(gold, prediction), "normalized_fallback")


class FinalAnswerMathBenchmark(BaseBenchmark):
    """One-stage local-data benchmark with native pass@k and resume support."""

    DEFAULT_DATA_FILE = ""
    EXPECTED_ROWS: Optional[int] = None
    REQUIRED_FIELDS = ("problem", "answer")
    ID_FIELD: Optional[str] = None
    GROUP_FIELD: Optional[str] = None
    USE_SOURCE_PROMPT = False
    PROMPT_TEMPLATE = DEFAULT_PROMPT
    PASS_TEMPERATURE = 0.7
    PASS_TOP_P = 1.0

    def __init__(
        self,
        data_file: Optional[str] = None,
        debug: bool = False,
        seed: Optional[List[int]] = None,
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
    ):
        super().__init__(
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
        )
        self.data_file = data_file or self.DEFAULT_DATA_FILE
        self.debug = debug
        self.seed = list(seed or [0, 1234, 1234, 1234])
        self.max_new_tokens = max_tokens

    def load_questions(self) -> List[Dict[str, Any]]:
        path = Path(self.data_file)
        with path.open("r", encoding="utf-8") as handle:
            questions = [json.loads(line) for line in handle if line.strip()]

        if path == Path(self.DEFAULT_DATA_FILE) and self.EXPECTED_ROWS is not None:
            if len(questions) != self.EXPECTED_ROWS:
                raise ValueError(f"{path} contains {len(questions)} rows; expected {self.EXPECTED_ROWS}")

        for row_idx, question in enumerate(questions):
            missing = [field for field in self.REQUIRED_FIELDS if field not in question]
            if missing:
                raise ValueError(f"{path} row {row_idx} is missing fields: {', '.join(missing)}")

        if self.ID_FIELD:
            ids = [str(question[self.ID_FIELD]) for question in questions]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path} contains duplicate {self.ID_FIELD} values")

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode enabled. Using only %s questions.", len(questions))
        self.logger.info("Loaded %s questions from %s", len(questions), path)
        return questions

    def prompt_for(self, example: Dict[str, Any]) -> str:
        if self.USE_SOURCE_PROMPT:
            return str(example["problem"])
        return self.PROMPT_TEMPLATE.format(problem=example["problem"])

    def extract_answer(self, output: str, example: Optional[Dict[str, Any]] = None) -> str:
        return extract_last_boxed(output)

    def grade_answer(self, gold: str, prediction: str, example: Dict[str, Any]) -> GradeOutcome:
        raise NotImplementedError

    def group_labels(self, example: Dict[str, Any]) -> Dict[str, str]:
        if not self.GROUP_FIELD:
            return {}
        return {self.GROUP_FIELD: str(example[self.GROUP_FIELD])}

    def _build_instances(
        self,
        model: LM,
        examples: List[Dict[str, Any]],
        do_sample: bool,
        seed: List[int],
        sample_idx: Optional[int] = None,
    ) -> List[Instance]:
        instances = []
        for idx, example in enumerate(examples):
            messages = [{"role": "user", "content": self.prompt_for(example)}]
            generation_kwargs: Dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.7,
                "seed": seed,
            }
            if do_sample:
                generation_kwargs["temperature"] = self.PASS_TEMPERATURE
                generation_kwargs["top_p"] = self.PASS_TOP_P
            instance = Instance(
                "generate_until",
                example,
                (self._prepare_messages(messages, model), generation_kwargs),
                idx,
            )
            if sample_idx is not None:
                instance.repeat_idx = sample_idx
            instances.append(instance)
        return instances

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()
        if self.num_samples > 1:
            return self._generate_pass_at_k(model, examples)

        instances = self._build_instances(model, examples, False, self.seed)
        self.logger.info("Generating responses for %s...", self.__class__.__name__)
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output, example)
        return {"examples": examples}

    def _generate_pass_at_k(self, model: LM, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        def build_instances(sample_idx: int, seed: List[int]) -> List[Instance]:
            return self._build_instances(model, examples, True, seed, sample_idx)

        self.logger.info(
            "Generating %s samples/problem for %s pass@k...",
            self.num_samples,
            self.__class__.__name__,
        )
        per_problem = self.generate_n_samples_batched(model, build_instances, self.num_samples)
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, per_problem):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(output, example) for output in outputs]
        return {"examples": examples, "pass_at_k": True}

    def _subgroup_metrics(
        self,
        examples: List[Dict[str, Any]],
        per_example_correct: List[Any],
        pass_at_k: bool,
    ) -> Dict[str, Any]:
        grouped: Dict[str, Dict[str, List[int]]] = {}
        for idx, example in enumerate(examples):
            for dimension, value in self.group_labels(example).items():
                grouped.setdefault(dimension, {}).setdefault(value, []).append(idx)

        metrics: Dict[str, Any] = {}
        for dimension, values in grouped.items():
            metrics[dimension] = {}
            for value, indices in sorted(values.items()):
                if pass_at_k:
                    counts = [int(per_example_correct[idx]) for idx in indices]
                    metrics[dimension][value] = {
                        "num_total": len(indices),
                        **self.aggregate_pass_at_k(counts),
                    }
                else:
                    solved = sum(bool(per_example_correct[idx]) for idx in indices)
                    metrics[dimension][value] = {
                        "num_total": len(indices),
                        "num_solved": solved,
                        "accuracy": solved / len(indices),
                    }
        return metrics

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        if not total:
            raise ValueError("Cannot evaluate an empty benchmark")

        if results.get("pass_at_k"):
            num_correct = []
            for example in examples:
                grades = [
                    self.grade_answer(str(example["answer"]), prediction, example)
                    for prediction in example["model_answers"]
                ]
                example["sample_grades"] = [asdict(grade) for grade in grades]
                num_correct.append(sum(int(grade.correct) for grade in grades))
            results.update(
                {
                    "num_total": total,
                    "num_samples": self.num_samples,
                    "num_correct": num_correct,
                    **self.aggregate_pass_at_k(num_correct),
                    "subgroups": self._subgroup_metrics(examples, num_correct, True),
                }
            )
            return results

        correct = []
        for example in examples:
            grade = self.grade_answer(str(example["answer"]), example["model_answer"], example)
            example["grade"] = asdict(grade)
            correct.append(grade.correct)
        solved = sum(correct)
        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
                "subgroups": self._subgroup_metrics(examples, correct, False),
            }
        )
        return results

    def _sample_prompt(self, example: Dict[str, Any]) -> str:
        return self.prompt_for(example)
