import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional  # noqa: F401

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .judge import judge_all

# Document-grounded prompt: the question is unanswerable without the supporting
# passage, so the full-page evidence from the source filing is rendered ahead of
# the question. The instruction mirrors the canonical FinanceBench framing -- a
# direct, concise answer grounded in the supplied document excerpt.
PROMPT = """Based on the following financial document excerpt, answer the question.

Document:
{evidence_text}

Question: {question}

Provide a direct, concise answer."""

# Default judge model when neither ``annotator_model`` nor ``$JUDGE_MODEL`` is supplied.
# ``gpt-4o-mini`` is the standard cheap-and-fast judge across the evalchemy LLM-judged
# benchmarks (HLE / MixEval / WildBench / MTBench).
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


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
    formatting or unit surface form (e.g. "$1,577M" vs "$1577.00"). See ``judge.py``.

    Link: https://github.com/patronus-ai/financebench
    """

    # FinanceBench's judge is an OpenAI chat model, so this benchmark is skipped at load
    # time when ``OPENAI_API_KEY`` is unset (TaskManager gates on this attribute).
    REQUIRES_OPENAI_ANNOTATOR = True

    def __init__(
        self,
        data_file: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "financebench.jsonl"
        ),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        annotator_model: Optional[str] = None,
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
            annotator_model: Override the judge model. Falls back to ``$JUDGE_MODEL`` and
                then ``gpt-4o-mini`` (the evalchemy-standard cheap judge).
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        # Resolution order matches the rest of evalchemy: explicit kwarg > env > default.
        self.judge_model = (
            annotator_model or os.environ.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
        )

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate answers using the provided model.

        Args:
            model: Language model.

        Returns:
            Dictionary containing the examples enriched with ``model_output``, or None
            for non-primary ranks.
        """
        examples = self.load_questions()

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

        self.logger.info(
            f"Judging {total} FinanceBench responses with {judge_model}..."
        )
        judgments = asyncio.run(judge_all(examples, judge_model))

        num_correct = 0
        num_incorrect = 0
        num_not_attempted = 0
        for example, (label, raw) in zip(examples, judgments):
            example["judge_label"] = label
            example["judge_raw"] = raw
            if label == "correct":
                num_correct += 1
            elif label == "not_attempted":
                num_not_attempted += 1
            else:
                num_incorrect += 1

        results.update(
            {
                "num_total": total,
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "num_not_attempted": num_not_attempted,
                "accuracy": num_correct / total if total else 0.0,
                "judge_model": judge_model,
            }
        )

        return results

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
