import hashlib
import logging
import os
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.aggregation import repeated_accuracy_results
from eval.contracts.sample_results import record_repeated_sample_metrics
from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for GPQADiamond benchmark"
    )


class GPQADiamondBenchmark(BaseBenchmark):
    """
    GPQADiamond Benchmark for evaluating multiple choice reasoning of LLMs.
    Link: https://huggingface.co/datasets/Idavidrein/gpqa
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize GPQADiamond benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "Idavidrein/gpqa"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 3

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and examples
        """
        examples = self.load_questions()
        for example in examples:
            multiple_choice_string, correct_answer = self.generate_multiple_choice_answers(example)
            example["multiple_choice_string"] = multiple_choice_string
            example["answer"] = correct_answer

        def build_instances(_repeat_idx: int) -> List[Instance]:
            all_instances = []
            for idx, example in enumerate(examples):
                messages = [
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            problem=example["Question"], options=example["multiple_choice_string"]
                        ),
                    },
                ]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "max_new_tokens": self.max_new_tokens,
                        },
                    ),
                    idx,
                )
                all_instances.append(instance)
            return all_instances

        self.logger.info("Generating seeded responses for GPQADiamond...")
        outputs_by_problem = self.generate_seeded_repeats(model, build_instances, self.n_repeat)
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, outputs_by_problem):
            example["model_outputs"] = outputs
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]

        return {"examples": examples}

    def grade_response(self, example: Dict[str, Any]) -> None:
        scores = [
            example["answer"] == answer
            for answer in example["model_answers"]
        ]
        record_repeated_sample_metrics(example, accuracy=scores)

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {**results, **repeated_accuracy_results(results["examples"])}

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GPQADiamond questions from the dataset."""
        dataset = load_dataset(self.dataset_name, "gpqa_diamond", cache_dir=HF_HUB_CACHE)
        questions = [row for row in dataset["train"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions

    def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Generate multiple choice string and correct answer letter."""
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        rnd = random.Random(42 + int(hashlib.md5(data["Question"].encode()).hexdigest(), 16))
        rnd.shuffle(answers)

        options = ["A", "B", "C", "D"]
        options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

        multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
        correct_answer_letter = next(
            letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter
