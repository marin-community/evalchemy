"""NUPA direct number-understanding and processing benchmark.

The source dataset is nested by task and digit length. This integration reads
that canonical JSON directly and reproduces the authors' text-model protocol:
100 deterministic examples from every task/digit group.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import ijson
from huggingface_hub import hf_hub_download
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.sample_results import record_sample_metrics
from eval.task import BaseBenchmark

from .scorer import ExampleScore, extract_answer, length_bucket, mean, score_prediction

SOURCE_DATASET_NAME = "HaotongYang/NUPA_text"
SOURCE_DATASET_REVISION = "01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7"
DEFAULT_SPLIT = "test"
DEFAULT_NUM_EACH = 100
DEFAULT_RANDOM_SEED = 20_222_943
BENCHMARK_SIZE = 238_926
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SMOKE_DATA = os.path.join(DATA_DIR, "nupa_smoke.jsonl")


class NUPABenchmark(BaseBenchmark):
    """Evaluate direct numeric answers using the NUPA text protocol."""

    METRICS = ("exact_match", "digit_match", "format_valid_rate")
    PRIMARY_METRIC = "exact_match"

    def __init__(
        self,
        dataset_name: str = SOURCE_DATASET_NAME,
        dataset_revision: str = SOURCE_DATASET_REVISION,
        dataset_split: str = DEFAULT_SPLIT,
        data_file: Optional[str] = None,
        num_each: int = DEFAULT_NUM_EACH,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_tokens: int = 256,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        if num_each <= 0:
            raise ValueError("num_each must be positive")
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.dataset_split = dataset_split
        self.data_file = data_file
        self.num_each = num_each
        self.random_seed = random_seed
        self.max_new_tokens = max_tokens
        self.debug = debug

    def benchmark_size(self) -> int | None:
        if self.debug:
            return _jsonl_size(Path(SMOKE_DATA))
        if self.data_file:
            return _jsonl_size(Path(self.data_file))
        if self.num_each == DEFAULT_NUM_EACH:
            return BENCHMARK_SIZE
        return None

    def validate_prepared_data(self) -> None:
        if self.debug or self.data_file:
            records = self._load_records()
            if not records:
                raise ValueError("NUPA data contains no records")
            for record in records:
                missing = _missing_record_fields(record)
                if missing:
                    raise ValueError(f"NUPA record {record.get('id', '<unknown>')} is missing {sorted(missing)}")

    def _load_records(self) -> List[Dict[str, Any]]:
        if self.debug:
            records: Iterable[Dict[str, Any]] = _read_jsonl(Path(SMOKE_DATA))
        elif self.data_file:
            records = _read_jsonl(Path(self.data_file))
        else:
            source = download_nupa_source(
                dataset_name=self.dataset_name,
                dataset_revision=self.dataset_revision,
                split=self.dataset_split,
            )
            records = iter_nupa_source_records(
                source,
                split=self.dataset_split,
                num_each=self.num_each,
                random_seed=self.random_seed,
            )
        return self.limit_samples(records)

    def _build_instances(self, model: LM, records: List[Dict[str, Any]]) -> List[Instance]:
        instances = []
        for idx, record in enumerate(records):
            messages = [{"role": "user", "content": record["prompt"]}]
            instances.append(
                Instance(
                    "generate_until",
                    record,
                    (
                        self._prepare_messages(messages, model),
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
        for record, output in zip(records, outputs, strict=True):
            example = dict(record)
            example["output"] = output
            examples.append(example)
        return {"examples": examples}

    def evaluate_responses(self, results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if results is None:
            return None

        scored_examples = []
        for example in results["examples"]:
            score = score_prediction(example.get("output"), example["answer"], example["answer_format"])
            example["model_answer"] = extract_answer(example.get("output"), example["answer_format"])
            example["correct"] = bool(score.exact_match)
            example["score"] = score
            record_sample_metrics(
                example,
                exact_match=score.exact_match,
                digit_match=score.digit_match,
                dlength=score.dlength,
                format_valid_rate=score.format_valid,
                no_answer_rate=score.no_answer,
            )
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

        results.update(metrics)
        results["dataset_num_samples"] = len(scored_examples)
        return results


def download_nupa_source(*, dataset_name: str, dataset_revision: str, split: str) -> Path:
    """Download and return one pinned NUPA source split."""
    return Path(
        hf_hub_download(
            repo_id=dataset_name,
            filename=f"{split}.json",
            repo_type="dataset",
            revision=dataset_revision,
        )
    )


def iter_nupa_source_records(
    source: Path,
    *,
    split: str,
    num_each: int = DEFAULT_NUM_EACH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Iterator[Dict[str, Any]]:
    """Yield the authors' deterministic text-evaluation sample from nested JSON."""
    if num_each <= 0:
        raise ValueError("num_each must be positive")
    random_generator = random.Random(random_seed)
    with source.open("rb") as source_file:
        for task_name, by_digit in ijson.kvitems(source_file, ""):
            digit_groups = _validate_digit_groups(task_name, by_digit)
            sampled_by_digit = {}
            for digit, examples in digit_groups.items():
                sampled_by_digit[str(digit)] = random_generator.sample(examples, min(num_each, len(examples)))
            yield from flatten_nupa_tasks({task_name: sampled_by_digit}, split=split)


def flatten_nupa_tasks(tasks: Mapping[str, Any], split: str) -> List[Dict[str, Any]]:
    """Flatten one or more NUPA task mappings into row-oriented records."""
    flattened: List[Dict[str, Any]] = []
    for task_name, by_digit in tasks.items():
        digit_groups = _validate_digit_groups(task_name, by_digit)
        answer_format = _answer_format_from_task_name(task_name)
        operation = _operation_from_task_name(task_name)
        max_digit = max(int(digit) for digit in digit_groups)
        if max_digit == 21:
            max_digit = 20
        for digit_key, examples in digit_groups.items():
            digit = int(digit_key)
            bucket = length_bucket(digit, max_digit=max_digit)
            for index, text in enumerate(examples):
                prompt, answer = split_prompt_answer(text)
                flattened.append(
                    {
                        "id": f"{split}:{task_name}:{digit}:{index:06d}",
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


def _validate_digit_groups(task_name: str, value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"NUPA task {task_name} must contain a digit mapping")
    digit_groups = {str(digit): examples for digit, examples in value.items() if isinstance(examples, list)}
    if len(digit_groups) != len(value):
        raise ValueError(f"NUPA task {task_name} has an invalid digit group")
    return digit_groups


def split_prompt_answer(text: str) -> tuple[str, str]:
    """Separate the source prompt from its answer delimiter."""
    if "=" not in text:
        raise ValueError("NUPA example has no answer delimiter")
    prompt, answer = text.split("=", 1)
    return f"{prompt.rstrip()} =", answer.strip()


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as data:
        return [json.loads(line) for line in data if line.strip()]


def _jsonl_size(path: Path) -> int:
    with path.open(encoding="utf-8") as data:
        return sum(bool(line.strip()) for line in data)


def _missing_record_fields(record: Mapping[str, Any]) -> set[str]:
    required = {"answer", "answer_format", "digit", "id", "length_bucket", "operation", "prompt", "task_name"}
    return required - record.keys()


def _operation_from_task_name(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) >= 2 and (parts[0] in {"digit", "get", "to"} or parts[1] in {"easy", "hard"}):
        return "_".join(parts[:2])
    return parts[0]


def _answer_format_from_task_name(task_name: str) -> str:
    answer_format = task_name.split("_")[-1]
    return "Integer" if answer_format == "int" else answer_format
