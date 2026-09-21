"""OpenAI SimpleQA benchmark."""

import ast
import asyncio
import csv
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import record_sample_metrics
from eval.contracts.task_outcome import TaskFailure
from eval.graders.answer_equivalence import EquivalenceJudgment, JudgeConfig, JudgeLabel
from eval.graders.simpleqa import SimpleQARequest, judge_simpleqa
from eval.robust_api import record_endpoint_failure
from eval.task import BaseBenchmark

DATASET_SIZE = 4_326
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "simple_qa_test_set.csv")


@dataclass(frozen=True)
class _GradeCounts:
    correct: int
    incorrect: int
    not_attempted: int
    judge_failed: int

    @property
    def judged(self) -> int:
        return self.correct + self.incorrect + self.not_attempted

    @property
    def attempted(self) -> int:
        return self.correct + self.incorrect


class SimpleQABenchmark(BaseBenchmark):
    """Evaluate short-form factual answers with OpenAI's SimpleQA classifier."""

    METRICS = ("accuracy", "accuracy_given_attempted", "f1")
    PRIMARY_METRIC = "accuracy"
    REQUIRES_JUDGE = True

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        debug: bool = False,
        seed: tuple[int, int, int, int] = (0, 1234, 1234, 1234),
        max_tokens: int = 512,
        annotator_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.judge_config = JudgeConfig.resolve(
            judge_model=annotator_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )
        self.judge_model = self.judge_config.model

    def benchmark_size(self) -> int:
        return DATASET_SIZE

    def validate_prepared_data(self) -> None:
        questions = self.load_questions()
        expected_size = min(2, self.benchmark_size()) if self.debug else self.benchmark_size()
        if len(questions) != expected_size:
            raise ValueError(f"SimpleQA expected {expected_size} questions, found {len(questions)}")
        if any(not example["question"].strip() or not example["answer"].strip() for example in questions):
            raise ValueError("SimpleQA questions and answers must not be empty")

    def _load_source_questions(self) -> List[Dict[str, Any]]:
        with open(self.data_file, newline="", encoding="utf-8") as data:
            questions = []
            for source_index, row in enumerate(csv.DictReader(data)):
                metadata = ast.literal_eval(row["metadata"])
                questions.append(
                    {
                        "source_index": source_index,
                        "question": row["problem"],
                        "answer": row["answer"],
                        "metadata": metadata,
                    }
                )
        return questions

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the pinned canonical CSV and retain its source row identity."""
        questions = self._load_source_questions()
        if self.debug:
            return questions[:2]
        return questions

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.limit_samples(self.load_questions())
        instances = []
        for example in examples:
            messages = [{"role": "user", "content": example["question"]}]
            instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        self._prepare_messages(messages, model),
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0,
                            "seed": self.seed,
                        },
                    ),
                    example["source_index"],
                )
            )

        self.logger.info("Generating %d SimpleQA responses...", len(instances))
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None
        for example, output in zip(examples, outputs, strict=True):
            example["model_output"] = output
        return {"examples": examples, "judge_model": self.judge_model}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if results is None:
            return None

        examples = results["examples"]
        judge_model = results.get("judge_model", self.judge_model)
        judgments = self._judge_examples(examples, judge_model)
        counts = self._record_judgments(examples, judgments)
        self._record_aggregate_metrics(results, counts, len(examples), judge_model)
        return results

    def _judge_examples(
        self,
        examples: List[Dict[str, Any]],
        judge_model: str,
    ) -> List[EquivalenceJudgment | BaseException]:
        judge_config = JudgeConfig(
            model=judge_model,
            base_url=self.judge_config.base_url,
            api_key=self.judge_config.api_key,
        )
        return asyncio.run(
            judge_simpleqa(
                [
                    SimpleQARequest(
                        question=example["question"],
                        target=example["answer"],
                        predicted_answer=example.get("model_output", "") or "",
                    )
                    for example in examples
                ],
                judge_config,
            )
        )

    def _record_judgments(
        self,
        examples: List[Dict[str, Any]],
        judgments: List[EquivalenceJudgment | BaseException],
    ) -> _GradeCounts:
        counts = {label: 0 for label in JudgeLabel}
        num_judge_failed = 0
        for index, (example, judgment) in enumerate(zip(examples, judgments, strict=True)):
            if isinstance(judgment, BaseException):
                if not isinstance(judgment, Exception):
                    raise judgment
                num_judge_failed += 1
                self._record_judge_failure(example, index, judgment)
                continue

            assert isinstance(judgment, EquivalenceJudgment)
            label = judgment.label
            counts[label] += 1
            self._record_judgment(example, judgment)
        return _GradeCounts(
            correct=counts[JudgeLabel.CORRECT],
            incorrect=counts[JudgeLabel.INCORRECT],
            not_attempted=counts[JudgeLabel.NOT_ATTEMPTED],
            judge_failed=num_judge_failed,
        )

    def _record_judge_failure(self, example: Dict[str, Any], index: int, error: Exception) -> None:
        record_endpoint_failure(FailureCategory.GRADER_INFRASTRUCTURE)
        example["judge_label"] = None
        example["judge_raw"] = None
        example["failure_category"] = FailureCategory.GRADER_INFRASTRUCTURE.value
        example["judge_error"] = asdict(
            TaskFailure(
                category=FailureCategory.GRADER_INFRASTRUCTURE,
                message=str(error)[:512] or type(error).__name__,
                exception_type=type(error).__name__,
            )
        )
        record_sample_metrics(example, judge_failed=True)
        self.logger.warning("SimpleQA judge failed for trial %d: %s", index, error)

    @staticmethod
    def _record_judgment(example: Dict[str, Any], judgment: EquivalenceJudgment) -> None:
        label = judgment.label
        example["judge_label"] = label.value
        example["judge_raw"] = judgment.raw
        record_sample_metrics(
            example,
            accuracy=label == JudgeLabel.CORRECT,
            incorrect=label == JudgeLabel.INCORRECT,
            not_attempted=label == JudgeLabel.NOT_ATTEMPTED,
            judge_failed=False,
        )

    @staticmethod
    def _record_aggregate_metrics(
        results: Dict[str, Any],
        counts: _GradeCounts,
        total: int,
        judge_model: str,
    ) -> None:
        accuracy = counts.correct / counts.judged if counts.judged else None
        accuracy_given_attempted = counts.correct / counts.attempted if counts.attempted else 0.0
        f1 = (
            2 * accuracy * accuracy_given_attempted / (accuracy + accuracy_given_attempted)
            if accuracy is not None and accuracy + accuracy_given_attempted
            else 0.0
        )
        results.update(
            {
                "num_total": total,
                "num_judged": counts.judged,
                "num_judge_failed": counts.judge_failed,
                "num_correct": counts.correct,
                "num_incorrect": counts.incorrect,
                "num_not_attempted": counts.not_attempted,
                "accuracy": accuracy,
                "correct_rate": accuracy,
                "incorrect_rate": counts.incorrect / counts.judged if counts.judged else None,
                "not_attempted_rate": counts.not_attempted / counts.judged if counts.judged else None,
                "accuracy_given_attempted": accuracy_given_attempted,
                "f1": f1,
                "judge_coverage": counts.judged / total if total else 0.0,
                "judge_model": judge_model,
            }
        )

    def to_samples(self, generation_result: Dict[str, Any], scored_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        samples = super().to_samples(generation_result, scored_result)
        for sample, example in zip(samples, generation_result["examples"], strict=True):
            sample["judge_label"] = example["judge_label"]
            sample["judge_raw"] = example["judge_raw"]
            if "failure_category" in example:
                sample["failure_category"] = example["failure_category"]
                sample["judge_error"] = example["judge_error"]
        return samples

    def _sample_doc(self, example: Dict[str, Any]) -> Dict[str, Any]:
        doc = super()._sample_doc(example)
        for key in ("judge_label", "judge_raw", "failure_category", "judge_error"):
            doc.pop(key, None)
        return doc
