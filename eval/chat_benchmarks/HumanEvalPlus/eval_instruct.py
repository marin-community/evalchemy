from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from .human_eval_plus.evaluation import evaluate_functional_correctness
from .utils.utils import extract_generation_code, language_settings
from eval.contracts.grading import GenerationArtifactManifest, GraderExecutionMode, generation_artifacts
from eval.task import BaseBenchmark


class HumanEvalPlusBenchmark(BaseBenchmark):
    """
    HumanEvalPlus benchmark for evaluating code generation capabilities across different languages.
    """

    GRADER_EXECUTION_MODE = GraderExecutionMode.SANDBOXED

    def __init__(
        self,
        languages: List[str] = ["python"],
        data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        max_tokens: int = 1024,
        num_workers: int = 8,
        timeout: float = 3.0,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HumanEvalPlus benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing HumanEvalPlus datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.languages = languages
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug

    def build_deepseekcoder_instruction(self, language: str, question: str) -> str:
        """Build instruction prompt for the model."""
        return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
            language.lower(), question.strip()
        )

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        results = {}
        artifacts = GenerationArtifactManifest.temporary()

        for lang in self.languages:
            try:
                problem_file = os.path.join(self.data_dir, f"humanevalplus-{lang}.jsonl")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                examples = [json.loads(x) for x in open(problem_file) if x.strip()]
                self.logger.info(f"Loaded {len(examples)} examples for {lang}")

                if self.debug:
                    examples = examples[:2]
                    self.logger.info("Debug mode: using first 2 examples only")

                all_instances = []
                for idx, example in enumerate(examples):
                    prompt = self.build_deepseekcoder_instruction(
                        language_settings[lang]["full_name"], example["prompt"]
                    )
                    inputs = self._prepare_messages([{"role": "user", "content": prompt}], model)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                self.logger.info("Generating responses for Human Eval Plus...")
                outputs = self.compute(model, all_instances, sample_namespace=lang)

                if model.rank != 0:
                    continue

                generated_examples = []
                for example, output in zip(examples, outputs):
                    example_with_output = example.copy()
                    example_with_output["output"] = output
                    processed_example = extract_generation_code(example_with_output, lang_code=lang)
                    generated_examples.append(processed_example)

                results[lang] = generated_examples
                artifacts.write_jsonl(
                    f"generated-{lang}",
                    f"generated_{lang}.jsonl",
                    generated_examples,
                    expected_count=len(examples),
                )

                self.logger.info(f"Generated and saved {len(generated_examples)} examples for {lang}")

            except Exception as e:
                self.logger.error(f"Error processing language {lang}: {str(e)}")
                continue

        results["examples"] = [
            {**example, "language": language} for language in self.languages for example in results.get(language, [])
        ]
        results["artifacts"] = artifacts
        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated code completions.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        artifacts = results["artifacts"]
        artifacts.validate_required()
        temp_dir = str(artifacts.root)

        evaluation_results = {}
        scored_count = 0

        for lang in self.languages:
            problem_file = os.path.join(self.data_dir, f"humanevalplus-{lang}.jsonl")
            temp_file_path = str(artifacts.path(f"generated-{lang}"))

            result = evaluate_functional_correctness(
                input_file=temp_file_path,
                tmp_dir=temp_dir,
                n_workers=self.num_workers,
                timeout=self.timeout,
                problem_file=problem_file,
                language=lang,
            )

            for metric, value in result.items():
                if metric == "scored_count":
                    scored_count += value
                else:
                    evaluation_results[f"{lang}_{metric}"] = value

            self.logger.info(f"Completed evaluation for {lang}")

        evaluation_results["scored_count"] = scored_count
        return evaluation_results

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation results, or None for non-primary ranks
        """
        self.logger.info(f"Running HumanEvalPlus benchmark for languages: {self.languages}")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            try:
                return self.evaluate_responses(generation_results)
            finally:
                artifacts = generation_artifacts(generation_results)
                if artifacts is not None:
                    artifacts.cleanup()
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
