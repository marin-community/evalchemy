import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.chat_benchmarks.amc_utils import amc_answer_is_correct, extract_boxed_answer
from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AMCBenchmark(BaseBenchmark):
    """Shared implementation for AMC text-only reasoning benchmarks."""

    TASK_NAME = "AMC"
    DATA_FILE = ""

    def __init__(
        self,
        data_file: Optional[str] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file or self.DATA_FILE
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 10

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()
        all_outputs = []

        for repeat_idx in range(self.n_repeat):
            seed = [s + repeat_idx for s in self.seed]
            instances = []
            for idx, example in enumerate(examples):
                prompt = PROMPT.format(problem=self.render_question(example))
                messages = [{"role": "user", "content": prompt}]
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
                instance.metadata = {
                    "problem_id": str(example.get("id", idx)),
                    "expected_answer": str(example["answer"]),
                    "year": example.get("year"),
                    "exam": example.get("exam"),
                    "problem_number": example.get("problem_number"),
                }
                instances.append(instance)

            self.logger.info(f"Generating responses for {self.TASK_NAME}...")
            all_outputs.append(self.compute(model, instances))

        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [
                self.extract_answer(output) for output in outputs
            ]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)
        all_results = []
        for repeat_idx in range(self.n_repeat):
            solved = sum(
                amc_answer_is_correct(
                    example["answer"],
                    example["model_answers"][repeat_idx],
                    example.get("accepted_answers"),
                )
                for example in examples
            )
            all_results.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": np.mean([result["num_solved"] for result in all_results]),
                "run_stats": all_results,
                "accuracy_avg": np.mean([result["accuracy"] for result in all_results]),
                "accuracy_std_err": np.std(
                    [result["accuracy"] for result in all_results]
                )
                / np.sqrt(self.n_repeat),
                "num_repeat": self.n_repeat,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        with open(self.data_file, "r") as f:
            questions = [json.loads(line) for line in f if line.strip()]

        if self.debug:
            questions = questions[:2]
            self.logger.info(
                f"Debug mode enabled. Using only {len(questions)} questions."
            )

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def render_question(self, example: Dict[str, Any]) -> str:
        question = str(example["question"]).strip()
        choices = example.get("choices")
        if not choices:
            return question

        choice_lines = []
        for label in sorted(choices):
            choice_lines.append(f"({label}) {choices[label]}")
        return f"{question}\n\nChoices:\n" + "\n".join(choice_lines)

    def extract_answer(self, output: str) -> str:
        return extract_boxed_answer(output)

    def _sample_prompt(self, example: Dict[str, Any]) -> str:
        return PROMPT.format(problem=self.render_question(example))
