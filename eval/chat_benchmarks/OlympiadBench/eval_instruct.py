import asyncio
import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import record_sample_metrics
from eval.contracts.task_outcome import TaskFailure
from eval.graders.answer_extraction import (
    AnswerExtractionError,
    ExtractionFailure,
    extract_boxed_answer,
    extraction_failure,
)
from eval.graders.answer_equivalence import (
    EquivalenceMethod,
    EquivalenceRequest,
    EquivalenceResult,
    JudgeConfig,
    grade_math_equivalence,
)
from eval.generation_stops import END_OF_TURN_SEQUENCES
from eval.robust_api import record_endpoint_failure
from eval.task import BaseBenchmark

# Same prompt shape as MATH500/AMC23/AIME24 (math reasoning benchmarks in this tree):
# the explicit "Mark your solution with \boxed" instruction makes answer extraction reliable.
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""

# This is the hand-selected 30-example stratified export introduced in
# https://github.com/marin-community/evalchemy/pull/24. It came from
# lmms-lab/olympiadbench[test_en], but that export recorded neither an immutable
# dataset revision nor the selection script. Keep it for historical comparison
# only; `OlympiadBenchFull` is the reproducible benchmark.
DEFAULT_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "olympiadbench.jsonl"
)
DEFAULT_DATASET = "lmms-lab/olympiadbench"
DEFAULT_SPLIT = "test_en"


def _strip_dollars(s: str) -> str:
    """Strip a single pair of surrounding ``$...$`` (or ``$$...$$``) delimiters if present."""
    s = s.strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        inner = s[1:-1].strip()
        if inner.startswith("$") and inner.endswith("$") and len(inner) >= 2:
            inner = inner[1:-1].strip()
        return inner
    return s


def _split_multiple_answers(s: str) -> List[str]:
    """Split a single reference-answer string that may itself carry several answers.

    OlympiadBench packs multiple answers into one ``final_answer`` list entry as
    ``$...$, $...$`` (each wrapped in dollars). Strip the dollars first, then split
    on commas that separate top-level dollar groups. Falls back to the whole string.
    """
    cleaned = _strip_dollars(s)
    if "$" in s:
        # Re-split on the original dollar-delimited groups, then strip each.
        groups = re.findall(r"\$([^$]*)\$", s)
        if groups:
            return [g.strip() for g in groups if g.strip()]
    if cleaned:
        return [cleaned]
    return []


def _flatten_reference_answers(raw: Any) -> List[str]:
    """Normalize the stored ``answer`` field into a flat list of candidate gold strings.

    Accepts a list of strings (the OlympiadBench ``final_answer`` shape, where each
    entry may itself contain several dollar-delimited answers) and returns the union of
    the individual candidate answers. Strings and single-string inputs are also accepted.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        raw = [str(raw)]
    out: List[str] = []
    for item in raw:
        out.extend(_split_multiple_answers(str(item)))
    return out


class OlympiadBenchBenchmark(BaseBenchmark):
    """
    OlympiadBench Benchmark for evaluating competition math/physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/lmms-lab/olympiadbench

    Uses the text-only English split (``test_en``). Each problem's reference answer may
    be a single value or a list of acceptable values; grading is correct-if-any-match
    through Minerva/SymPy first and an LLM equivalence judge for unresolved answers.
    Follows the MATH500/AMC23 pattern in this tree: the model is asked to box its
    final answer.
    """

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    REQUIRES_JUDGE = True

    def benchmark_size(self) -> int:
        return len(self.load_questions())

    def benchmark_metrics(self):
        if self.num_samples > 1:
            return super().benchmark_metrics()
        return ("accuracy_avg",) if self.n_repeat > 1 else self.METRICS

    def benchmark_primary_metric(self):
        if self.num_samples > 1:
            return super().benchmark_primary_metric()
        return "accuracy_avg" if self.n_repeat > 1 else self.PRIMARY_METRIC

    def __init__(
        self,
        data_file: Optional[str] = DEFAULT_DATA_FILE,
        dataset_name: str = DEFAULT_DATASET,
        dataset_revision: Optional[str] = None,
        dataset_split: str = DEFAULT_SPLIT,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
        n_repeat: int = 10,
        annotator_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
    ):
        """
        Initialize OlympiadBench benchmark.

        Args:
            data_file: Local JSONL containing the legacy 30-example subset.
            dataset_name: Source dataset identifier retained with the subset provenance.
            dataset_revision: Source revision when loading a reproducible derived task.
            dataset_split: Source dataset split.
            debug: If set, only evaluate on 2 examples.
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            max_tokens: Max generation tokens. These are hard olympiad problems; default 32768.
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
            num_samples: Number of completions per problem. 1 (default) = single-sample path.
            pass_at_k: k-list for pass@k aggregation (only used when num_samples > 1).
            n_repeat: Number of seeded repetitions for the subset's single-sample score.
            annotator_model: Explicit judge model, or ``auto`` to use ``$JUDGE_MODEL``.
            judge_api_key: Judge credential. Falls back to ``$JUDGE_API_KEY``.
            judge_base_url: Judge endpoint. Falls back to ``$JUDGE_BASE_URL``.
        """
        super().__init__(
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
        )
        self.data_file = data_file
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.dataset_split = dataset_split
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = n_repeat
        self.judge_config = JudgeConfig.resolve(
            judge_model=annotator_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()

        # ---- native pass@k path: num_samples > 1 ----
        if self.num_samples > 1:
            return self._generate_pass_at_k(model, examples)

        if self.n_repeat > 1:
            return self._generate_repeated_responses(model, examples)

        # Prepare instances for model
        all_instances = []
        for idx, example in enumerate(examples):
            messages = [
                {"role": "user", "content": PROMPT.format(problem=example["problem"])},
            ]

            templated_messages = self._prepare_messages(messages, model)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": self.seed,
                            "until": list(END_OF_TURN_SEQUENCES),
                        },
                    ),
                    idx,
                )
            )

        # Generate model responses
        self.logger.info("Generating responses for OlympiadBench...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"], example["answer_extraction_error"] = self._extract_for_scoring(output)

        return {"examples": examples}

    def _generate_repeated_responses(
        self, model: LM, examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate one deterministic completion per problem for each seeded repetition."""
        all_outputs = []
        for repeat_idx in range(self.n_repeat):
            seed = [value + repeat_idx for value in self.seed]
            all_instances = []
            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["problem"])},
                ]
                templated_messages = self._prepare_messages(messages, model)
                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                            "until": list(END_OF_TURN_SEQUENCES),
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = repeat_idx
                all_instances.append(instance)

            self.logger.info("Generating repeated responses for OlympiadBench...")
            all_outputs.append(self.compute(model, all_instances))

        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            extracted = [self._extract_for_scoring(output) for output in outputs]
            example["model_answers"] = [answer for answer, _ in extracted]
            example["answer_extraction_errors"] = [error for _, error in extracted]
        return {"examples": examples}

    def _generate_pass_at_k(
        self, model: LM, examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate ``num_samples`` completions per problem via the base scaffold."""

        def build_instances(sample_idx: int, seed: List[int]) -> List[Instance]:
            instances = []
            for idx, example in enumerate(examples):
                messages = [
                    {
                        "role": "user",
                        "content": PROMPT.format(problem=example["problem"]),
                    }
                ]
                templated_messages = self._prepare_messages(messages, model)
                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            # Sampling on (pass@k needs diversity); seed varies per sample.
                            "do_sample": True,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "top_p": 1.0,
                            "seed": seed,
                            "until": list(END_OF_TURN_SEQUENCES),
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = sample_idx
                instances.append(instance)
            return instances

        self.logger.info(
            f"Generating {self.num_samples} samples/problem for OlympiadBench pass@k..."
        )
        per_problem = self.generate_n_samples_batched(
            model, build_instances, self.num_samples
        )
        if model.rank != 0:
            return None
        for example, outputs in zip(examples, per_problem):
            example["model_outputs"] = list(outputs)
            extracted = [self._extract_for_scoring(output) for output in outputs]
            example["model_answers"] = [answer for answer, _ in extracted]
            example["answer_extraction_errors"] = [error for _, error in extracted]
        return {"examples": examples, "pass_at_k": True}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)

        # ---- native pass@k aggregation ----
        if results.get("pass_at_k"):
            correct_by_example, grading_summary = self._grade_model_answers(
                examples, [example["model_answers"] for example in examples]
            )
            num_correct = [sum(int(correct) for correct in answers) for answers in correct_by_example]
            self.record_pass_at_k_metrics(examples, num_correct)
            pass_at_k_table = self.aggregate_pass_at_k(num_correct)
            results.update(
                {
                    "num_total": total,
                    "num_samples": self.num_samples,
                    "num_correct": num_correct,
                    **grading_summary,
                    **self._dataset_provenance(total),
                    **pass_at_k_table,
                }
            )
            return results

        if self.n_repeat > 1:
            correct_by_example, grading_summary = self._grade_model_answers(
                examples, [example["model_answers"] for example in examples]
            )
            all_results = []
            correct_by_repeat = [
                [answers[repeat_idx] for answers in correct_by_example]
                for repeat_idx in range(self.n_repeat)
            ]
            for repeat_idx in range(self.n_repeat):
                solved = sum(correct_by_repeat[repeat_idx])
                all_results.append(
                    {
                        "repetition": repeat_idx + 1,
                        "num_total": total,
                        "num_solved": solved,
                        "accuracy": solved / total,
                    }
                )

            self.record_repeated_accuracy(examples, correct_by_repeat)
            accuracies = [result["accuracy"] for result in all_results]
            results.update(
                {
                    "num_total": total,
                    "solved_avg": np.mean([result["num_solved"] for result in all_results]),
                    "run_stats": all_results,
                    "accuracy_avg": np.mean(accuracies),
                    "accuracy_std_err": np.std(accuracies) / np.sqrt(self.n_repeat),
                    "num_repeat": self.n_repeat,
                    **grading_summary,
                    **self._dataset_provenance(total),
                }
            )
            return results

        correct_by_example, grading_summary = self._grade_model_answers(
            examples, [[example["model_answer"]] for example in examples]
        )
        correct = [answers[0] for answers in correct_by_example]
        for example, is_correct in zip(examples, correct):
            record_sample_metrics(example, accuracy=is_correct)
        solved = sum(correct)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
                "accuracy_stderr": np.sqrt((solved / total) * (1 - solved / total) / (total - 1))
                if total > 1
                else 0.0,
                **grading_summary,
                **self._dataset_provenance(total),
            }
        )

        return results

    def _grade_model_answers(
        self,
        examples: List[Dict[str, Any]],
        answers_by_example: List[List[str]],
    ) -> tuple[List[List[bool]], Dict[str, Any]]:
        """Grade extracted answers with Minerva followed by the shared LLM judge."""
        requests = []
        positions = []
        for example_index, (example, answers) in enumerate(zip(examples, answers_by_example, strict=True)):
            references = tuple(_flatten_reference_answers(example["answer"]))
            for answer_index, answer in enumerate(answers):
                requests.append(
                    EquivalenceRequest(
                        question=str(example.get("problem", example.get("question", ""))),
                        reference_answers=references,
                        candidate_answer=answer,
                    )
                )
                positions.append((example_index, answer_index))

        outcomes = asyncio.run(grade_math_equivalence(requests, self.judge_config))
        correct_by_example = [[False] * len(answers) for answers in answers_by_example]
        grades_by_example: List[List[Dict[str, Any]]] = [[{} for _ in answers] for answers in answers_by_example]
        num_graded_by_minerva = 0
        num_judged_by_llm = 0
        num_judge_failed = 0
        for (example_index, answer_index), outcome in zip(positions, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome
                num_judge_failed += 1
                record_endpoint_failure(FailureCategory.GRADER_INFRASTRUCTURE)
                grades_by_example[example_index][answer_index] = {
                    "method": EquivalenceMethod.LLM_JUDGE.value,
                    "judge_label": None,
                    "judge_raw": None,
                    "judge_error": asdict(
                        TaskFailure(
                            category=FailureCategory.GRADER_INFRASTRUCTURE,
                            message=str(outcome)[:512] or type(outcome).__name__,
                            exception_type=type(outcome).__name__,
                        )
                    ),
                }
                continue

            assert isinstance(outcome, EquivalenceResult)
            correct_by_example[example_index][answer_index] = outcome.equivalent
            if outcome.method == EquivalenceMethod.MINERVA:
                num_graded_by_minerva += 1
                grades_by_example[example_index][answer_index] = {
                    "method": outcome.method.value,
                    "judge_label": None,
                    "judge_raw": None,
                }
                continue

            num_judged_by_llm += 1
            assert outcome.judgment is not None
            grades_by_example[example_index][answer_index] = {
                "method": outcome.method.value,
                "judge_label": outcome.judgment.label.value,
                "judge_raw": outcome.judgment.raw,
            }

        for example, grades in zip(examples, grades_by_example, strict=True):
            example["equivalence_grades"] = grades
            if len(grades) == 1:
                example.update(grades[0])

        return correct_by_example, {
            "num_graded_by_minerva": num_graded_by_minerva,
            "num_judged_by_llm": num_judged_by_llm,
            "num_judge_failed": num_judge_failed,
            "judge_model": self.judge_config.model,
        }

    def to_samples(self, generation_result: Dict[str, Any], scored_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        samples = super().to_samples(generation_result, scored_result)
        for sample, example in zip(samples, generation_result["examples"], strict=True):
            sample["equivalence_grades"] = example["equivalence_grades"]
        return samples

    def _sample_doc(self, example: Dict[str, Any]) -> Dict[str, Any]:
        doc = super()._sample_doc(example)
        for key in ("equivalence_grades", "method", "judge_label", "judge_raw", "judge_error"):
            doc.pop(key, None)
        return doc

    def _dataset_provenance(self, total: int) -> Dict[str, Any]:
        """Return source details persisted with each score artifact."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "dataset_split": self.dataset_split,
            "dataset_num_samples": total,
        }

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the bundled legacy OlympiadBench subset."""
        assert self.data_file is not None
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")

        if self.debug:
            questions = questions[:2]
            self.logger.info(
                f"Debug mode enabled. Using only {len(questions)} questions."
            )

        return questions

    def _load_from_hf(self) -> List[Dict[str, Any]]:
        """Project the HF dataset into the local JSONL record shape."""
        from datasets import load_dataset

        cache_dir = os.environ.get("HF_HUB_CACHE")
        ds = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            revision=self.dataset_revision,
            cache_dir=cache_dir,
        )

        out: List[Dict[str, Any]] = []
        for ex in ds:
            source = ex.get("source") or ""
            if "_TO_" not in source:
                continue
            final_answer = ex.get("final_answer")
            if not final_answer:
                continue
            subject = (
                "mathematics"
                if "maths" in source
                else ("physics" if "physics" in source else "unknown")
            )
            context = ex.get("context")
            question = ex.get("question") or ""
            out.append(
                {
                    "id": ex.get("question_id"),
                    "problem": (context + "\n\n" if context else "") + question,
                    "question": question,
                    "context": context,
                    "answer": list(final_answer),
                    "subject": subject,
                    "subfield": ex.get("subfield"),
                    "unit": ex.get("unit"),
                    "answer_type": ex.get("answer_type"),
                    "is_multiple_answer": bool(ex.get("is_multiple_answer")),
                    "error": ex.get("error"),
                    "source": source,
                }
            )
        return out

    def extract_answer(self, output: str) -> str:
        """Extract the first boxed answer before any later task boundary.

        Raises:
            EmptyResponseError: The model response is empty.
            MissingAnswerError: No complete boxed answer precedes the boundary.
        """
        return extract_boxed_answer(output)

    def _extract_for_scoring(self, output: str) -> tuple[str, Optional[ExtractionFailure]]:
        """Score missing syntax as incorrect while retaining a structured error."""
        try:
            return self.extract_answer(output), None
        except AnswerExtractionError as exc:
            return "", extraction_failure(exc)
