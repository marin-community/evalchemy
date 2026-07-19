import logging
import math
import os
from collections import defaultdict
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

try:
    from .auto_scoring_judge import AutoScoringJudge
except ImportError:  # TaskManager executes this module with the benchmark directory on sys.path.
    from auto_scoring_judge import AutoScoringJudge


# Adapted from the official English open-ended prompt at:
# https://github.com/OpenBMB/OlympiadBench/blob/ba5b26a7e2849940b598a9159c1190daa2b9175f/inference/code/evaluators/evaluator.py#L29-L101
PROMPT = """The following is an open-ended problem from an International {subject} competition. {answer_requirement}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {answer_format}." and give the result explicitly{unit_instruction}.
{question_content}"""

DEFAULT_DATASET = "Hothan/OlympiadBench"
DEFAULT_DATASET_REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"
DEFAULT_SPLIT = "train"
BENCHMARK_SCOPE = "english_open_ended_text_only"
DEFAULT_PRECISION = 1e-8

# The English open-ended text-only scope comprises exactly these two subsets.
ENGLISH_TEXT_SUBSETS: Dict[str, Tuple[str, int]] = {
    "OE_TO_maths_en_COMP": ("Math", 674),
    "OE_TO_physics_en_COMP": ("Physics", 236),
}

ENGLISH_ANSWER_TYPE_TEXT = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


class OlympiadBenchBenchmark(BaseBenchmark):
    """English open-ended text-only OlympiadBench evaluation."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        dataset_revision: str = DEFAULT_DATASET_REVISION,
        dataset_split: str = DEFAULT_SPLIT,
        cache_dir: Optional[str] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
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
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.dataset_split = dataset_split
        self.cache_dir = cache_dir if cache_dir is not None else os.environ.get("HF_HUB_CACHE")
        self.subsets = list(ENGLISH_TEXT_SUBSETS)
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.judge = AutoScoringJudge()

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()
        if self.num_samples > 1:
            return self._generate_pass_at_k(model, examples)

        instances = [
            self._make_instance(model, example, idx, do_sample=False, seed=self.seed)
            for idx, example in enumerate(examples)
        ]
        self.logger.info("Generating responses for OlympiadBench...")
        outputs = self.compute(model, instances)

        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)

        return {"examples": examples, **self._result_metadata()}

    def _generate_pass_at_k(self, model: LM, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        def build_instances(sample_idx: int, seed: List[int]) -> List[Instance]:
            instances = []
            for idx, example in enumerate(examples):
                instance = self._make_instance(model, example, idx, do_sample=True, seed=seed)
                instance.repeat_idx = sample_idx
                instances.append(instance)
            return instances

        self.logger.info(
            "Generating %s samples/problem for OlympiadBench pass@k...",
            self.num_samples,
        )
        per_problem = self.generate_n_samples_batched(model, build_instances, self.num_samples)
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, per_problem):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(output) for output in outputs]

        return {
            "examples": examples,
            "pass_at_k": True,
            **self._result_metadata(),
        }

    def _make_instance(
        self,
        model: LM,
        example: Dict[str, Any],
        idx: int,
        do_sample: bool,
        seed: List[int],
    ) -> Instance:
        messages = [{"role": "user", "content": self._build_prompt(example)}]
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7 if do_sample else 0.0,
            "seed": seed,
        }
        if do_sample:
            generation_kwargs["top_p"] = 1.0

        instance = Instance(
            "generate_until",
            example,
            (self._prepare_messages(messages, model), generation_kwargs),
            idx,
        )
        instance.metadata = {
            "problem_id": str(example["id"]),
            "subset": example["subset"],
            "subject": example["subject"],
            "subfield": example["subfield"],
            "answer_type": example["answer_type"],
            "expected_answers": list(example["answer"]),
            "is_multiple_answer": example["is_multiple_answer"],
            "unit": example["unit"],
            "error": "" if example["error"] is None else example["error"],
        }
        return instance

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if results is None:
            return None

        examples = results["examples"]
        if results.get("pass_at_k"):
            num_correct = []
            for example in examples:
                correctness = [self._score_output(example, output) for output in example["model_outputs"]]
                example["sample_correctness"] = correctness
                num_correct.append(sum(int(value) for value in correctness))

            results.update(
                {
                    "num_total": len(examples),
                    "num_samples": self.num_samples,
                    "num_correct": num_correct,
                    **self.aggregate_pass_at_k(num_correct),
                }
            )
            self._add_subject_pass_at_k(results, examples, num_correct)
            return results

        for example in examples:
            example["is_correct"] = self._score_output(example, example.get("model_output", ""))

        solved = sum(int(example["is_correct"]) for example in examples)
        results.update(
            {
                "num_total": len(examples),
                "num_solved": solved,
                "accuracy": solved / len(examples) if examples else 0.0,
            }
        )
        self._add_subject_accuracy(results, examples)
        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        questions = []
        canonical_dataset = (
            self.dataset_name == DEFAULT_DATASET
            and self.dataset_revision == DEFAULT_DATASET_REVISION
            and self.dataset_split == DEFAULT_SPLIT
        )

        for subset in self.subsets:
            dataset = load_dataset(
                self.dataset_name,
                subset,
                split=self.dataset_split,
                revision=self.dataset_revision,
                cache_dir=self.cache_dir,
            )
            expected_count = ENGLISH_TEXT_SUBSETS[subset][1]
            if canonical_dataset and not self.debug and len(dataset) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} rows in {subset} at revision "
                    f"{self.dataset_revision}, found {len(dataset)}."
                )

            rows = islice(dataset, 2) if self.debug else dataset
            questions.extend(self._normalize_example(dict(row), subset) for row in rows)

        if not questions:
            raise ValueError("OlympiadBench did not load any English text-only questions.")

        self.logger.info(
            "Loaded %s English text-only OlympiadBench questions from %s@%s.",
            len(questions),
            self.dataset_name,
            self.dataset_revision,
        )
        return questions

    def _normalize_example(self, row: Dict[str, Any], subset: str) -> Dict[str, Any]:
        problem_id = row.get("id", "")
        if problem_id in (None, ""):
            raise ValueError(f"OlympiadBench row in {subset} has no problem ID.")
        expected_subject = ENGLISH_TEXT_SUBSETS[subset][0]
        expected_fields = {
            "modality": "Text-only",
            "question_type": "Open-ended",
            "language": "English",
            "subject": expected_subject,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Unexpected {field}={row.get(field)!r} for OlympiadBench "
                    f"problem {problem_id} in {subset}; expected {expected!r}."
                )

        if any(value for key, value in row.items() if key.startswith("image_")):
            raise ValueError(f"Text-only OlympiadBench problem {problem_id} unexpectedly contains an image.")

        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"OlympiadBench problem {problem_id} has no question text.")

        raw_answers = row.get("final_answer")
        if not isinstance(raw_answers, list):
            raise ValueError(f"OlympiadBench problem {problem_id} has a non-list final_answer.")
        answers = []
        for answer in raw_answers:
            if answer is None:
                continue
            # Dollar signs are presentation delimiters. Removing them also repairs the
            # two canonical rows that mix delimited and undelimited answer components.
            normalized_answer = str(answer).replace("$", "").strip()
            if normalized_answer:
                answers.append(normalized_answer)
        if not answers:
            raise ValueError(f"OlympiadBench problem {problem_id} has no final answer.")

        answer_type = str(row.get("answer_type") or "").strip()
        if not answer_type:
            raise ValueError(f"OlympiadBench problem {problem_id} has no answer type.")
        answer_types = [item.strip() for item in answer_type.split(",") if item.strip()]
        if not answer_types:
            raise ValueError(f"OlympiadBench problem {problem_id} has an invalid answer type.")

        context = str(row.get("context") or "").strip()
        return {
            "id": problem_id,
            "subset": subset,
            "problem": f"{context}\n\n{question}" if context else question,
            "question": question,
            "context": context,
            "answer": answers,
            # Problem 1482 has a composite Equation,Numerical answer despite a false
            # source flag. The answer-type schema is authoritative for that case.
            "is_multiple_answer": bool(row.get("is_multiple_answer")) or len(answer_types) > 1,
            "unit": str(row.get("unit") or ""),
            "answer_type": answer_type,
            "error": row.get("error"),
            "difficulty": str(row.get("difficulty") or ""),
            "subfield": str(row.get("subfield") or ""),
            "subject": expected_subject,
        }

    def _build_prompt(self, example: Dict[str, Any]) -> str:
        context = example.get("context", "")
        answer_types = [item.strip() for item in example["answer_type"].split(",") if item.strip()]
        answer_type_texts = [self._answer_type_text(item) for item in answer_types]
        if not example["is_multiple_answer"]:
            answer_requirement = f"The answer of The problem should be {answer_type_texts[0]}. "
            answer_format = r"\boxed{answer}"
        else:
            answer_format = r"\boxed{multiple answers connected with commas}"
            if len(set(answer_type_texts)) == 1:
                answer_requirement = (
                    f"The problem has multiple answers, each of them should be {answer_type_texts[0]}. "
                )
            else:
                answer_type_list = ", ".join(answer_type_texts)
                answer_requirement = (
                    f"The problem has multiple answers, with the answers in order being {answer_type_list}. "
                )

        unit_instruction = ""
        if example["unit"]:
            answer_format += "(unit)"
            unit_instruction = r", note that the unit of the answer should not be included in \boxed{}"

        question_content = f"{context}\n{example['question']}" if context else example["question"]
        return PROMPT.format(
            subject=example["subject"],
            answer_requirement=answer_requirement,
            answer_format=answer_format,
            unit_instruction=unit_instruction,
            question_content=question_content,
        )

    def _answer_type_text(self, answer_type: str) -> str:
        for name, text in ENGLISH_ANSWER_TYPE_TEXT.items():
            if name in answer_type:
                return text
        raise ValueError(f"Unsupported OlympiadBench answer type: {answer_type!r}.")

    def _sample_prompt(self, example: Dict[str, Any]) -> str:
        return self._build_prompt(example)

    def _score_output(self, example: Dict[str, Any], output: str) -> bool:
        if not str(output).strip():
            return False

        precision = self._parse_precision(example.get("error"))
        return any(self.judge.judge(candidate, output, precision=precision) for candidate in example["answer"])

    def _parse_precision(self, error: Any) -> Union[float, List[float]]:
        if isinstance(error, (list, tuple)):
            values = [self._precision_value(item) for item in error]
            return values or DEFAULT_PRECISION
        if isinstance(error, str) and "," in error:
            return [self._precision_value(item) for item in error.split(",")]
        return self._precision_value(error)

    def _precision_value(self, value: Any) -> float:
        if value in (None, "", "null"):
            return DEFAULT_PRECISION
        try:
            precision = float(value)
        except (TypeError, ValueError):
            return DEFAULT_PRECISION
        if not math.isfinite(precision) or precision < 0:
            return DEFAULT_PRECISION
        return precision

    def _add_subject_accuracy(self, results: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
        grouped = defaultdict(list)
        for example in examples:
            grouped[example["subject"]].append(int(example["is_correct"]))

        for subject, scores in grouped.items():
            results[f"num_total_subject_{subject}"] = len(scores)
            results[f"num_solved_subject_{subject}"] = sum(scores)
            results[f"accuracy_subject_{subject}"] = sum(scores) / len(scores)

    def _add_subject_pass_at_k(
        self,
        results: Dict[str, Any],
        examples: List[Dict[str, Any]],
        num_correct: List[int],
    ) -> None:
        grouped = defaultdict(list)
        for example, correct in zip(examples, num_correct):
            grouped[example["subject"]].append(correct)

        for subject, correct_counts in grouped.items():
            results[f"num_total_subject_{subject}"] = len(correct_counts)
            for metric, value in self.aggregate_pass_at_k(correct_counts).items():
                results[f"{metric}_subject_{subject}"] = value

    def _result_metadata(self) -> Dict[str, Any]:
        return {
            "benchmark_scope": BENCHMARK_SCOPE,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "subsets": list(self.subsets),
        }

    def extract_answer(self, output: str) -> str:
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer if answer is not None else ""
        except Exception:
            return ""
