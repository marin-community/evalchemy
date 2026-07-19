"""IFBench: instruction-following eval against the real allenai/IFBench_test set.

IFBench is AI2's harder successor to IFEval -- 300 WildChat-sourced prompts each
carrying one or more *verifiable* constraints drawn from 58 distinct types
(``count:keywords_multiple``, ``format:emoji``, ``ratio:stop_words``,
``words:palindrome``, ...). Constraints are checked programmatically by the
canonical AI2 graders vendored alongside this file (``instructions.py`` etc.),
so the reported accuracy is directly comparable to the IFBench paper's.

This benchmark mirrors the IFEval chat-benchmark structure (load → generate →
evaluate), but pulls its data from HuggingFace rather than a vendored JSONL.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .grader import evaluate_accuracy


class IFBenchBenchmark(BaseBenchmark):
    """Instruction-following benchmark over ``allenai/IFBench_test``."""

    def __init__(
        self,
        num_examples: Optional[int] = None,
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """Initialize the IFBench instruction-following benchmark.

        Args:
            num_examples: Cap on the number of examples to evaluate (None = all 300).
            debug: If True, only evaluate 2 examples.
            max_tokens: Maximum number of tokens for generation.
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.num_examples = num_examples
        self.debug = debug
        self.max_tokens = max_tokens

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the IFBench prompts from the ``allenai/IFBench_test`` HF dataset.

        Each row's ``prompt`` already carries the constraint inline as natural
        language (e.g. ``"... Include exactly 5 numbers in the response."``); the
        machine-checkable constraint metadata lives alongside it in
        ``instruction_id_list`` and ``kwargs`` and is consumed by the grader.
        """
        dataset: Dataset = load_dataset("allenai/IFBench_test", split="train")
        self.logger.info("Loaded %d examples from allenai/IFBench_test", len(dataset))

        questions: List[Dict[str, Any]] = []
        for ex in dataset:
            # Coerce numpy/pandas scalar types to native Python so the row is
            # JSON-serializable when we later write the response file.
            questions.append(
                {
                    "key": int(ex["key"]) if ex["key"] is not None else len(questions),
                    "prompt": ex["prompt"],
                    "instruction_id_list": list(ex["instruction_id_list"]),
                    "kwargs": [dict(kw) for kw in ex["kwargs"]],
                }
            )

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode: using 2 examples")
        elif self.num_examples is not None:
            questions = questions[: self.num_examples]

        return questions

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses to each IFBench prompt using the provided model.

        Args:
            model: Language model instance.

        Returns:
            Dictionary carrying the temp dir holding the written responses plus counts,
            or ``None`` on non-primary ranks.
        """
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        questions = self.load_questions()
        self.logger.info("Processing %d examples", len(questions))

        all_instances = []
        for idx, question in enumerate(questions):
            inputs = self._prepare_messages(
                [{"role": "user", "content": question["prompt"]}], model
            )
            all_instances.append(
                Instance(
                    "generate_until",
                    question,
                    (
                        inputs,
                        {
                            "max_new_tokens": self.max_tokens,
                            "do_sample": False,
                            "temperature": 0,
                        },
                    ),
                    idx,
                )
            )

        self.logger.info("Generating responses...")
        outputs = self.compute(model, all_instances)

        if model.rank != 0:
            return None

        generated = []
        for question, output in zip(questions, outputs):
            item = question.copy()
            item["response"] = output
            generated.append(item)

        output_path = os.path.join(temp_dir, "ifbench.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in generated:
                f.write(json.dumps(item) + "\n")

        return {
            "temp_dir_obj": temp_dir_obj,
            "output_path": output_path,
            "num_examples": len(generated),
            "total_examples": len(questions),
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generated responses by checking each verifiable constraint.

        Reports strict + loose variants of prompt- and instruction-level accuracy
        (the IFBench paper headlines prompt-level loose), plus a per-constraint-type
        breakdown. See ``grader.py`` for the scoring protocol.

        Args:
            results: Dictionary carrying the generation temp dir and counts.

        Returns:
            Dictionary containing the evaluation metrics.
        """
        temp_dir_obj = results["temp_dir_obj"]
        try:
            result = evaluate_accuracy(results["output_path"])
            result.update(
                {
                    "num_examples": results["num_examples"],
                    "completion_rate": results["num_examples"]
                    / results["total_examples"],
                }
            )
            return result
        finally:
            temp_dir_obj.cleanup()

    def run_benchmark(self, model: LM) -> Dict[str, Any]:
        """Run the complete IFBench evaluation pipeline.

        Args:
            model: Language model instance.

        Returns:
            Dictionary containing the evaluation metrics.
        """
        self.logger.info("Starting IFBench evaluation")
        generation_results = self.generate_responses(model)
        evaluation_results = self.evaluate_responses(generation_results)
        evaluation_results.update(
            {
                "benchmark_version": "ifbench",
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
        )
        return evaluation_results
