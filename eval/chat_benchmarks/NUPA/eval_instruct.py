"""NUPA: direct number understanding and processing evaluation.

NUPA is the text Q&A benchmark from "Number Cookbook: Number Understanding of
Language Models and How to Improve It". This native Evalchemy benchmark treats
NUPA as one Evalchemy task and reports metrics over flattened prompt/answer
examples, with NUPA task-family and length-bucket breakdowns.

Design: marin-community/marin#7297.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .scorer import ExampleScore, extract_answer, length_bucket, mean, normalize_answer, score_prediction

SOURCE_DATASET_NAME = "HaotongYang/NUPA_text"
# TODO(marin-community/marin#7297): Finalize the owning Hugging Face organization
# and repository name before merging the NUPA integration.
PUBLISHED_DATASET_NAME = "TODO_ORG/nupa-text-eval"
DATASET_NAME = PUBLISHED_DATASET_NAME
DEFAULT_SPLIT = "test"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SMOKE_DATA = os.path.join(DATA_DIR, "nupa_smoke.jsonl")
FLAT_RECORD_FIELDS = {
    "answer",
    "answer_format",
    "digit",
    "id",
    "length_bucket",
    "operation",
    "prompt",
    "task_name",
}


class NUPABenchmark(BaseBenchmark):
    """Native Evalchemy wrapper for NUPA direct numeric QA."""

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        dataset_split: str = DEFAULT_SPLIT,
        data_file: Optional[str] = None,
        max_tokens: int = 256,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.data_file = data_file
        self.max_new_tokens = max_tokens
        self.debug = debug

    def _load_records(self) -> List[Dict[str, Any]]:
        if self.debug:
            records = _read_jsonl(SMOKE_DATA)
        elif self.data_file:
            records = _read_jsonl(self.data_file)
        else:
            records = _load_flattened_hf_records(self.dataset_name, self.dataset_split)
        return records

    def _build_instances(self, model: LM, records: List[Dict[str, Any]]) -> List[Instance]:
        instances = []
        for idx, record in enumerate(records):
            prompt = record["prompt"]
            messages = [{"role": "user", "content": prompt}]
            templated = self._prepare_messages(messages, model)
            instances.append(
                Instance(
                    "generate_until",
                    record,
                    (
                        templated,
                        {"do_sample": False, "temperature": 0.0, "max_new_tokens": self.max_new_tokens},
                    ),
                    idx,
                )
            )
        return instances

    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        records = self._load_records()
        self.logger.info("Generating responses for NUPA (%d examples)...", len(records))
        outputs = self.compute(model, self._build_instances(model, records))
        if model.rank != 0:
            return None
        examples = []
        for record, output in zip(records, outputs):
            example = dict(record)
            example["output"] = output
            examples.append(example)
        return {"examples": examples}

    def evaluate_responses(self, results: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if results is None:
            return None

        scored_examples = []
        for example in results["examples"]:
            score = score_prediction(example.get("output"), example["answer"], example["answer_format"])
            example["model_answer"] = extract_answer(example.get("output"), example["answer_format"])
            example["normalized_model_answer"] = normalize_answer(example["model_answer"], example["answer_format"])
            example["correct"] = bool(score.exact_match)
            example["score"] = score
            scored_examples.append(example)

        metrics: Dict[str, float] = {}
        metrics.update(_aggregate_scores(scored_examples, prefix=""))
        for group_name, items in _group_by(scored_examples, "task_name").items():
            metrics.update(_aggregate_scores(items, prefix=f"task:{group_name}/"))
        for group_name, items in _group_by(scored_examples, "length_bucket").items():
            metrics.update(_aggregate_scores(items, prefix=f"bucket:{group_name}/"))
        for task_name, task_items in _group_by(scored_examples, "task_name").items():
            for bucket, bucket_items in _group_by(task_items, "length_bucket").items():
                metrics.update(_aggregate_scores(bucket_items, prefix=f"task:{task_name}/bucket:{bucket}/"))

        metrics["dataset_num_samples"] = float(len(scored_examples))
        return metrics


def _aggregate_scores(examples: List[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    scores: List[ExampleScore] = [example["score"] for example in examples]
    return {
        f"{prefix}exact_match": mean(score.exact_match for score in scores),
        f"{prefix}digit_match": mean(score.digit_match for score in scores),
        f"{prefix}dlength": mean(score.dlength for score in scores),
        f"{prefix}format_valid_rate": mean(score.format_valid for score in scores),
        f"{prefix}no_answer_rate": mean(score.no_answer for score in scores),
    }


def _group_by(examples: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[str(example[key])].append(example)
    return dict(groups)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_flattened_hf_records(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    records: List[Dict[str, Any]] = []
    for row in dataset:
        record = dict(row)
        if FLAT_RECORD_FIELDS.issubset(record):
            records.append(record)
        else:
            records.extend(flatten_nupa_row(record, split=split))
    return records


def flatten_nupa_row(row: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Flatten one row from the auto-converted HF NUPA_text dataset."""
    flattened: List[Dict[str, Any]] = []
    for task_name, by_digit in row.items():
        if not isinstance(by_digit, dict):
            continue
        answer_format = _answer_format_from_task_name(task_name)
        operation = _operation_from_task_name(task_name)
        max_digit = max(int(digit_key) for digit_key, examples in by_digit.items() if isinstance(examples, list))
        for digit_key, examples in by_digit.items():
            if not isinstance(examples, list):
                continue
            digit = int(digit_key)
            bucket = length_bucket(digit, max_digit=max_digit)
            for idx, text in enumerate(examples):
                prompt, answer = split_prompt_answer(text)
                flattened.append(
                    {
                        "id": f"{split}:{task_name}:{digit}:{idx:06d}",
                        "task_name": task_name,
                        "operation": operation,
                        "answer_format": answer_format,
                        "digit": digit,
                        "length_bucket": bucket,
                        "prompt": prompt,
                        "answer": answer,
                    }
                )
    return flattened


def split_prompt_answer(text: str) -> tuple[str, str]:
    prompt, answer = text.split("=", 1)
    return f"{prompt.rstrip()} =", answer.strip()


def _operation_from_task_name(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) >= 2 and parts[0] == "multiply" and parts[1] in {"easy", "hard"}:
        return "_".join(parts[:2])
    if len(parts) >= 2 and parts[0] in {"digit", "to"}:
        return "_".join(parts[:2])
    return parts[0]


def _answer_format_from_task_name(task_name: str) -> str:
    answer_format = task_name.split("_")[-1]
    return "Integer" if answer_format == "int" else answer_format
