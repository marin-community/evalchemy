import asyncio
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional  # noqa: F401

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import record_sample_metrics
from eval.contracts.task_outcome import TaskFailure
from eval.graders.answer_equivalence import (
    EquivalenceJudgment,
    EquivalenceRequest,
    JudgeConfig,
    JudgeLabel,
    judge_equivalence,
)
from eval.robust_api import record_endpoint_failure
from eval.task import BaseBenchmark

# Document-grounded prompt: the question is unanswerable without the supporting
# passage, so the full-page evidence from the source filing is rendered ahead of
# the question. The instruction mirrors the canonical FinanceBench framing -- a
# direct, concise answer grounded in the supplied document excerpt.
PROMPT = """Based on the following financial document excerpt, answer the question.

Document:
{evidence_text}

Question: {question}

Provide a direct, concise answer."""

class FinanceBenchBenchmark(BaseBenchmark):
    """
    FinanceBench Benchmark for evaluating financial document Q&A.

    FinanceBench (PatronusAI/financebench) tests whether an LLM can answer questions that
    require reading and reasoning over financial documents -- 10-K / 10-Q filings and
    earnings-call transcripts -- spanning numerical, boolean, and summary question types.

    Each question is shipped with the supporting passage from the source filing, which is
    injected into the prompt as document context so the answer is grounded in the text
    rather than recalled from parametric memory. Without that context the benchmark is
    unsolvable: the questions are about specific line items in specific filings.

    Grading is LLM-as-judge (SimpleQA-style correct / incorrect / not_attempted) rather
    than exact match, because financial answers frequently differ from the gold only in
    formatting or unit surface form (e.g. "$1,577M" vs "$1577.00"). The shared
    answer-equivalence judge uses ``JUDGE_API_KEY`` and optional ``JUDGE_BASE_URL``
    credentials that are separate from the candidate endpoint's ``OPENAI_API_KEY``.

    Link: https://github.com/patronus-ai/financebench
    """

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    REQUIRES_JUDGE = True

    def __init__(
        self,
        data_file: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "financebench.jsonl"
        ),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        annotator_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize FinanceBench benchmark.

        Args:
            data_file: JSONL file of FinanceBench items
                (id, question, answer, evidence_text, doc_name, company, question_type).
            debug: If set, only evaluate on 2 examples.
            seed: Random seed for reproducibility (deterministic at temperature 0).
            max_tokens: Max generation tokens. 4096 by default -- factual Q&A answers
                are short, but the prompt's document context can be long.
            annotator_model: Override the judge model. The CLI's ``auto`` sentinel is
                treated as unset, then falls back to ``$JUDGE_MODEL`` and finally
                ``gpt-4o-mini`` (the evalchemy-standard cheap judge).
            judge_api_key: Judge credential. Falls back to ``$JUDGE_API_KEY``.
            judge_base_url: OpenAI-compatible judge endpoint. Falls back to
                ``$JUDGE_BASE_URL`` and then the OpenAI API.
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
        """
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

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate answers using the provided model.

        Args:
            model: Language model.

        Returns:
            Dictionary containing the examples enriched with ``model_output``, or None
            for non-primary ranks.
        """
        examples = self.limit_samples(self.load_questions())

        all_instances = []
        for idx, example in enumerate(examples):
            content = PROMPT.format(
                question=example["question"],
                evidence_text=example["evidence_text"],
            )
            messages = [{"role": "user", "content": content}]
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
                            # Deterministic decoding -- FinanceBench is factual Q&A, so a
                            # single greedy answer is the right evaluation signal.
                            "temperature": 0,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
            )

        self.logger.info("Generating responses for FinanceBench...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks.
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output

        return {"examples": examples, "judge_model": self.judge_model}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Grade the generated answers with the LLM judge and aggregate accuracy."""
        # Handle None result from non-primary ranks.
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        judge_model = results.get("judge_model", self.judge_model)
        judge_config = JudgeConfig(
            model=judge_model,
            base_url=self.judge_config.base_url,
            api_key=self.judge_config.api_key,
        )

        self.logger.info(
            f"Judging {total} FinanceBench responses with {judge_model}..."
        )
        judgments = asyncio.run(
            judge_equivalence(
                [
                    EquivalenceRequest(
                        question=example["question"],
                        reference_answers=(str(example["answer"]),),
                        candidate_answer=example.get("model_output", "") or "",
                    )
                    for example in examples
                ],
                judge_config,
            )
        )

        num_correct = 0
        num_incorrect = 0
        num_not_attempted = 0
        num_judge_failed = 0
        for index, (example, judgment) in enumerate(zip(examples, judgments, strict=True)):
            if isinstance(judgment, BaseException):
                if not isinstance(judgment, Exception):
                    raise judgment
                num_judge_failed += 1
                record_endpoint_failure(FailureCategory.GRADER_INFRASTRUCTURE)
                example["judge_label"] = None
                example["judge_raw"] = None
                example["failure_category"] = FailureCategory.GRADER_INFRASTRUCTURE.value
                example["judge_error"] = asdict(
                    TaskFailure(
                        category=FailureCategory.GRADER_INFRASTRUCTURE,
                        message=str(judgment)[:512] or type(judgment).__name__,
                        exception_type=type(judgment).__name__,
                    )
                )
                record_sample_metrics(example, judge_failed=True)
                self.logger.warning("FinanceBench judge failed for trial %d: %s", index, judgment)
                continue

            assert isinstance(judgment, EquivalenceJudgment)
            label = judgment.label
            example["judge_label"] = label.value
            example["judge_raw"] = judgment.raw
            record_sample_metrics(
                example,
                accuracy=label == JudgeLabel.CORRECT,
                not_attempted=label == JudgeLabel.NOT_ATTEMPTED,
                judge_failed=False,
            )
            if label == JudgeLabel.CORRECT:
                num_correct += 1
            elif label == JudgeLabel.NOT_ATTEMPTED:
                num_not_attempted += 1
            else:
                num_incorrect += 1

        num_judged = total - num_judge_failed
        results.update(
            {
                "num_total": total,
                "num_judged": num_judged,
                "num_judge_failed": num_judge_failed,
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "num_not_attempted": num_not_attempted,
                "accuracy": num_correct / num_judged if num_judged else None,
                "judge_coverage": num_judged / total if total else 0.0,
                "judge_model": judge_model,
            }
        )

        return results

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

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load FinanceBench questions from the local data file.

        The shipped ``data/financebench.jsonl`` is a 50-item deterministic sample of the
        open-source slice of ``patronus-ai/financebench``
        (``data/financebench_open_source.jsonl``, 150 rows). Each item carries the
        ``evidence_text`` -- the full page of the source filing the question is grounded
        in -- so the benchmark runs offline the same way MATH500 / AIME24 do.
        """
        with open(self.data_file, "r") as f:
            questions = [json.loads(line) for line in f if line.strip()]

        if self.debug:
            questions = questions[:2]
            self.logger.info(
                f"Debug mode enabled. Using only {len(questions)} questions."
            )

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions
