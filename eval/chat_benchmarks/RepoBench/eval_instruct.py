from typing import Dict, List, Any, Optional
import tempfile
import json
import os
import logging
from tqdm import tqdm
from datasets import load_dataset
from fuzzywuzzy import fuzz
from itertools import islice

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from archive_data.utils import load_data, construct_trainable_data, get_first_line_not_comment
from data.utils import construct_prompt
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
from eval.task import BaseBenchmark
from eval.contracts.sample_results import record_sample_metrics


class RepoBenchmark(BaseBenchmark):
    """
    RepoBench benchmark implementation evaluating code completion capabilities
    across different programming languages and contexts.
    """

    def __init__(
        self,
        languages: List[str] = ["python", "java"],
        subsets: List[str] = ["cross_file_first", "cross_file_random", "in_file"],
        max_tokens: int = 128,
        debug: bool = False,
        legacy_mode: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize RepoBench benchmark.

        Args:
            languages: List of programming languages to evaluate
            subsets: List of dataset subsets to use
            max_tokens: Maximum number of tokens for generation
            debug: If true, run on debug mode using 2 samples
            legacy_mode: Whether to use legacy (v0) evaluation
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.languages = languages
        self.subsets = subsets
        self.max_tokens = max_tokens
        self.debug = debug
        self.legacy_mode = legacy_mode

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing temporary directory with generated responses,
            or None for non-primary ranks
        """
        if self.legacy_mode:
            return self._generate_responses_legacy(model)

        if model.rank == 0:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name
            results = {"temp_dir_obj": temp_dir_obj}

        for lang in self.languages:
            datasets = load_dataset(f"tianyang/repobench_{lang}_v1.1", verification_mode="no_checks")

            for subset, dataset in datasets.items():
                if subset not in self.subsets:
                    continue

                if self.debug:
                    dataset = dataset.select(range(2))
                    self.logger.info(f"Debug mode: using 2 examples")

                all_instances = []
                # Split dataset across ranks for parallel construction
                # Get subset of dataset for this rank using built-in slice functionality
                rank_dataset = list(
                    islice(enumerate(dataset), model.rank, len(dataset), model.world_size)
                )

                # Process examples for this rank's shard
                for idx, example in rank_dataset:
                    prompt = construct_prompt(
                        example, tokenizer=model.tokenizer, max_token_nums=self.max_tokens, language=lang
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                prompt,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "temperature": 0.2,
                                    "top_p": 0.95,
                                    "do_sample": True,
                                },
                            ),
                            idx,
                        )
                    )
                    all_instances[-1].sample_ordinal = idx
                self.logger.info("Generating responses for RepoBench...")
                outputs = self.compute(
                    model,
                    all_instances,
                    do_slice=False,
                    sample_namespace=f"{lang}:{subset}",
                )

                # Only rank 0 should save the results
                if model.rank != 0:
                    continue

                generated_examples = []
                for idx, (example, output) in enumerate(zip(dataset, outputs)):
                    generated_examples.append(
                        {
                            "idx": idx,
                            "model_output": output,
                            "gpt_completion": get_first_line_not_comment(output, language=lang),
                            "label": example["next_line"],
                        }
                    )

                results[f"{lang}:{subset}"] = generated_examples
                output_path = f"{temp_dir}/repobench_{subset}_{lang}.jsonl"
                with open(output_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

        if model.rank == 0:
            return results

    def _generate_responses_legacy(self, model: LM) -> Dict[str, Any]:
        """Legacy (v0) generation implementation."""
        prefix_token = "<fim_prefix>"
        suffix_token = "<fim_suffix><fim_middle>"

        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        results = {"temp_dir_obj": temp_dir_obj}

        for lang in self.languages:
            for subset in self.subsets:
                dataset = load_data(split="test", task="completion", language=lang, length="2k", setting=subset)

                examples = construct_trainable_data(dataset, language=lang)

                all_instances = []
                for idx, example in enumerate(examples):
                    prompt = example["data"]

                    if "star" in model._model.config._name_or_path:
                        prompt = prefix_token + prompt + suffix_token

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                prompt,
                                {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.95, "do_sample": True},
                            ),
                            idx,
                        )
                    )

                outputs = self.compute(
                    model,
                    all_instances,
                    sample_namespace=f"legacy:{lang}:{subset}",
                )

                if model.rank != 0:
                    continue

                generated_examples = []
                for idx, (example, output) in enumerate(zip(examples, outputs)):
                    generated_examples.append(
                        {
                            "idx": idx,
                            "prompt": example["data"],
                            "model_output": output,
                            "gpt_completion": get_first_line_not_comment(output, language=lang),
                            "label": example["label"],
                        }
                    )

                results[f"legacy:{lang}:{subset}"] = generated_examples
                output_path = f"{temp_dir}/repobench_{subset}_{lang}.jsonl"
                with open(output_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

        return results

    def sample_groups(self, generation_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        prefix = "legacy:" if self.legacy_mode else ""
        return {
            f"{prefix}{lang}:{subset}": generation_result[f"{prefix}{lang}:{subset}"]
            for lang in self.languages for subset in self.subsets
            if f"{prefix}{lang}:{subset}" in generation_result
        }

    def grade_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        try:
            for examples in self.sample_groups(results).values():
                for example in examples:
                    record_sample_metrics(
                        example,
                        exact_match=example["label"].split() == example["gpt_completion"].split(),
                        edit_similarity=fuzz.ratio(example["label"], example["gpt_completion"]),
                    )
            return results
        finally:
            results["temp_dir_obj"].cleanup()

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        groups = self.sample_groups(results)
        scores = {
            namespace: {
                "EM": round(round(sum(row["sample_metrics"]["exact_match"] for row in examples) / len(examples), 5) * 100, 2),
                "ES": round(round(sum(row["sample_metrics"]["edit_similarity"] for row in examples) / len(examples), 5), 2),
            }
            for namespace, examples in groups.items() if examples
        }
        metrics = {
            f"{namespace.removeprefix('legacy:').replace(':', '_')}_{metric}": scores[namespace][metric]
            for namespace in scores
            for metric in ("EM", "ES")
        }
        total = sum(len(groups[namespace]) for namespace in scores)
        if total:
            metrics.update({
                f"{lang}_weighted_avg_{metric}": round(
                    sum(scores[namespace][metric] * len(groups[namespace]) for namespace in scores) / total, 2
                )
                for lang in self.languages for metric in ("EM", "ES")
            })
        return metrics

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
