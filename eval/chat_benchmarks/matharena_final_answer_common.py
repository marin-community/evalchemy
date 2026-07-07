import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

_HMMT_DIR = Path(__file__).resolve().parent / "HMMT"
if str(_HMMT_DIR) not in sys.path:
    sys.path.insert(0, str(_HMMT_DIR))

from matharena.parser import check_answers, extract_answer, parse_answer  # noqa: E402

PROMPT = """Problem: {problem}
Please reason step by step, and put your final answer within \\boxed{{}}.
Answer:"""


class MathArenaFinalAnswerBenchmark(BaseBenchmark):
    """Shared implementation for MathArena final-answer competition datasets."""

    DATASET_NAME: str
    DATASET_REVISION: Optional[str] = None
    EXPECTED_NUM_ROWS: Optional[int] = None
    BENCHMARK_DESCRIPTION: str = ""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        debug: bool = False,
        max_tokens: int = 32768,
        seed: Optional[List[int]] = None,
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
        self.dataset_name = dataset_name or self.DATASET_NAME
        self.dataset_revision = dataset_revision or self.DATASET_REVISION
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed or [0, 1234, 1234, 1234]
        self.n_repeat = 10

    def _ensure_single_sample_mode(self) -> None:
        if self.num_samples != 1:
            raise ValueError(
                f"{self.__class__.__name__} does not implement native pass@k yet; "
                "run with --num_samples 1 or add a pass@k path for this benchmark."
            )

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate repeated final-answer completions with the HMMT-compatible MathArena protocol.

        Args:
            model: Language model.

        Returns:
            Dictionary containing generated responses, or None for non-primary ranks.
        """
        self._ensure_single_sample_mode()
        examples = self.load_questions()
        all_outputs = []

        for repeat_idx in range(self.n_repeat):
            all_instances = []
            seed = [s + repeat_idx for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [{"role": "user", "content": PROMPT.format(problem=example["problem"])}]
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
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = repeat_idx
                instance.metadata = self._metadata_for_example(example)
                all_instances.append(instance)

            self.logger.info(
                "Generating responses for %s repeat %s/%s...",
                self.BENCHMARK_DESCRIPTION or self.dataset_name,
                repeat_idx + 1,
                self.n_repeat,
            )
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            list_answer = "," in str(example["answer"])
            example["model_answers"] = [extract_answer(output, False, True, list_answer)[0] for output in outputs]
            example["label"] = []

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate generated completions using MathArena final-answer checking."""
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)
        all_results = []

        for repeat_idx in range(self.n_repeat):
            solved = 0
            for example in examples:
                gold_answer, _ = parse_answer(str(example["answer"]))
                model_answer = example["model_answers"][repeat_idx]
                is_correct = bool(check_answers(model_answer, gold_answer))
                example["label"].append(is_correct)
                solved += int(is_correct)
            accuracy = solved / num_questions
            all_results.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": accuracy,
                }
            )

        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load and normalize MathArena final-answer rows."""
        load_kwargs = {"split": "train"}
        if self.dataset_revision:
            load_kwargs["revision"] = self.dataset_revision

        dataset = load_dataset(self.dataset_name, **load_kwargs)
        questions = [self._normalize_example(dict(example), idx) for idx, example in enumerate(dataset)]

        if self.EXPECTED_NUM_ROWS is not None and not self.debug and len(questions) != self.EXPECTED_NUM_ROWS:
            raise ValueError(
                f"{self.dataset_name} expected {self.EXPECTED_NUM_ROWS} rows, got {len(questions)}"
            )

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode enabled. Using only %s questions.", len(questions))

        self.logger.info("Loaded %s questions from %s", len(questions), self.dataset_name)
        return questions

    def _normalize_example(self, example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        missing = {"problem", "answer"} - set(example)
        if missing:
            raise KeyError(f"{self.dataset_name} row {idx} is missing {sorted(missing)}")

        problem_id = example.get("problem_idx", example.get("id", idx))
        example["id"] = str(problem_id)
        example["answer"] = str(example["answer"])
        example["dataset_name"] = self.dataset_name
        return example

    def _metadata_for_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "problem_id": str(example["id"]),
            "expected_answer": str(example["answer"]),
            "dataset_name": self.dataset_name,
        }
        if "source" in example:
            metadata["source"] = example["source"]
        if "problem_type" in example:
            metadata["problem_type"] = example["problem_type"]
        return metadata
