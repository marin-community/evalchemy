from argparse import Namespace
from dataclasses import dataclass
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.chat_benchmarks.zeroeval.src.task_configs import prompt_generation
from eval.task import BaseBenchmark

from .src.unified_utils import mapping_task_names, save_outputs
from .src.evaluation.zebra_grid_eval import grade_model as zebra_grid_grade_model, aggregate_model as zebra_grid_aggregate_model, load_private_solutions
from .src.evaluation.crux_eval import grade_model as crux_grade_model, aggregate_model as crux_aggregate_model
from .src.evaluation.math_eval import grade_model as math_grade_model, aggregate_model as math_aggregate_model
from eval.contracts.sample_results import record_sample_metrics


@dataclass
class ZeroEvalConfig:
    # Dataset configuration
    start_index: int = 0
    end_index: int = -1


class ZeroEvalBenchmark(BaseBenchmark):
    """
    ZeroEval benchmark for a number of tasks and benchmarks.
    """

    def __init__(
        self,
        tasks: List[str] = ["zebra-grid", "numersense-v2", "crux", "math-l5"],
        config: Optional[ZeroEvalConfig] = None,
        max_tokens: Optional[int] = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger, system_instruction=system_instruction)
        self.tasks = tasks
        self.config = config or ZeroEvalConfig()
        self.max_new_tokens = int(max_tokens) if max_tokens is not None else 4096
        self.debug = debug

    def load_dataset(self, data_name: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset for ZeroEval tasks.
        """
        try:
            chat_history = []
            id_strs = []
            extracted_chats = []
            metadata = {}
            dataset, id_name = mapping_task_names(data_name)

            if self.debug:
                dataset = dataset.select(range(min(10, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            # Process each item
            prompt_generation_args = Namespace(run_name="")
            for ind, item in enumerate(dataset):
                id_strs.append(item.get(id_name, f"{data_name}#{ind}"))
                prompt = prompt_generation(data_name, item, prompt_generation_args)
                chat_history.append([prompt])
                extracted_chats.append([{"content": prompt, "role": "user"}])
                for key in item:
                    if key not in metadata:
                        metadata[key] = []
                    metadata[key].append(item[key])

            self.logger.info(f"Finished processing {data_name} dataset.")

            # Apply index limits
            if self.config.end_index < 0:
                self.config.end_index = len(id_strs)

            slice_range = slice(self.config.start_index, self.config.end_index)
            return (
                id_strs[slice_range],
                chat_history[slice_range],
                extracted_chats[slice_range],
                {k: v[slice_range] for k, v in metadata.items()},
            )

        except Exception as e:
            self.logger.error(f"Error loading dataset for task {data_name}: {e}")
            raise e

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for ZeroEval tasks.

        Args:
            model (LM): Language model instance

        Returns:
            Dict[str, Any]: Dictionary containing file paths and temporary directory
        """
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        results = {}

        for task in self.tasks:
            self.logger.info(f"Generating responses for task: {task}")

            # Load data
            try:
                id_strs, chat_history, extracted_chats, metadata = self.load_dataset(task)
            except Exception as e:
                self.logger.error(f"Error loading data for task {task}: {e}")
                continue

            self.logger.info(f"Finished loading data for task {task}.")

            # Apply template
            model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]

            output_path = os.path.join(temp_dir, f"{task}.json")

            # Generate responses
            self.logger.info("Generating responses for Zero Eval...")
            all_instances = [
                Instance(
                    "generate_until",
                    None,
                    (
                        inputs,
                        {
                            "temperature": 0.0,
                            "max_new_tokens": self.max_new_tokens,
                            "do_sample": False,
                        },
                    ),
                    idx,
                )
                for idx, inputs in enumerate(model_inputs)
            ]

            outputs = self.compute(model, all_instances, sample_namespace=task)

            if model.rank != 0:
                continue

            outputs = [[output] for output in outputs]

            # Save outputs
            save_args = Namespace(
                data_name=task,
                model_name="",
                engine="",
                repetition_penalty=0.0,
                temperature=0.0,
                top_p=0.0,
                max_tokens=self.max_new_tokens,
            )
            save_outputs(save_args, id_strs, outputs, chat_history, metadata, model_inputs, output_path)
            with open(output_path) as stream:
                results[task] = json.load(stream)

        results["temp_dir_obj"] = temp_dir_obj
        return results

    def sample_groups(self, generation_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        return {task: generation_result[task] for task in self.tasks if task in generation_result}

    def grade_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        graded = {}
        try:
            for task, examples in self.sample_groups(results).items():
                if task == "zebra-grid":
                    load_private_solutions()
                    rows = zebra_grid_grade_model("%", examples)
                else:
                    grader = math_grade_model if task in ("numersense-v2", "math-l5") else crux_grade_model
                    rows = grader("%", examples)
                for row in rows:
                    row["model_output"] = row["output"][0]
                    record_sample_metrics(row, accuracy=row["solved"] if task == "zebra-grid" else row["matched"] is True)
                graded[task] = rows
            return graded
        finally:
            results["temp_dir_obj"].cleanup()

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        for task, examples in self.sample_groups(results).items():
            reducer = (
                zebra_grid_aggregate_model if task == "zebra-grid"
                else math_aggregate_model if task in ("numersense-v2", "math-l5") else crux_aggregate_model
            )
            result = reducer("%", examples)
            metrics[task] = float(result["Puzzle Acc"] if task == "zebra-grid" else result["Acc"])
            if task == "zebra-grid":
                metrics[f"{task}_cell_acc"] = float(result["Cell Acc"])
            metrics[f"{task}_no_answer"] = float(result["No answer"])
            metrics[f"{task}_reason_lens"] = float(result["Reason Lens"])
        return metrics
