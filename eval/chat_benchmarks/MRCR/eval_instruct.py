"""OpenAI MRCR benchmark using the corrected, revision-pinned dataset."""

import json
import logging
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Union

import tiktoken
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.limits import (
    ContextWindowExceededError,
    encoded_token_count,
    message_content_token_count,
    require_context_length,
)
from eval.task import BaseBenchmark

DATASET_NAME = "openai/mrcr"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
DATA_FILES = (
    "2needle/2needle_0.parquet",
    "2needle/2needle_1.parquet",
    "4needle/4needle_0.parquet",
    "4needle/4needle_1.parquet",
    "8needle/8needle_0.parquet",
    "8needle/8needle_1.parquet",
)
MRCR_BIN_UPPER_BOUNDS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576)
MRCR_NEEDLE_COUNTS = (2, 4, 8)
_OFFICIAL_TOKENIZER = tiktoken.get_encoding("o200k_base")
PreparedPrompt = Union[List[Dict[str, str]], str]
SelectedExample = tuple[Dict[str, Any], PreparedPrompt]


def mrcr_bin(total_tokens: int) -> Optional[int]:
    """Return the published MRCR bin upper bound for a token count."""
    lower = 4_096
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower <= total_tokens <= upper:
            return upper
        lower = upper + 1
    return None


def parse_prompt(prompt: str) -> List[Dict[str, str]]:
    """Parse and validate the dataset's JSON-encoded chat messages."""
    messages = json.loads(prompt)
    if not isinstance(messages, list):
        raise ValueError(f"MRCR prompt must be a message list, got {type(messages).__name__}")
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("MRCR prompt messages must contain string role and content fields")
    return messages


def official_token_count(example: Dict[str, Any]) -> int:
    """Count prompt content and answer with OpenAI MRCR's o200k tokenizer."""
    messages = parse_prompt(str(example["prompt"]))
    return message_content_token_count(_OFFICIAL_TOKENIZER.encode, messages) + encoded_token_count(
        _OFFICIAL_TOKENIZER.encode, str(example["answer"])
    )


def score_response(response: str, answer: str, nonce: str) -> tuple[float, float]:
    """Return the official nonce-gated similarity and a prefix-hit indicator."""
    response = str(response)
    if not response.startswith(nonce):
        return 0.0, 0.0
    response_body = response.removeprefix(nonce)
    answer_body = str(answer).removeprefix(nonce)
    return float(SequenceMatcher(None, response_body, answer_body).ratio()), 1.0


def _interleave_cells(
    cells: Dict[tuple[int, int], List[SelectedExample]],
    desired_cells: List[tuple[int, int]],
    limit: Optional[int],
) -> List[SelectedExample]:
    ordered: List[SelectedExample] = []
    rounds = max((len(examples) for examples in cells.values()), default=0)
    for position in range(rounds):
        for cell in desired_cells:
            if position < len(cells[cell]):
                ordered.append(cells[cell][position])
                if limit is not None and len(ordered) == limit:
                    return ordered
    return ordered


class MRCRBenchmark(BaseBenchmark):
    """OpenAI's multi-round co-reference resolution long-context benchmark."""

    def __init__(
        self,
        n_needles: Optional[List[int]] = None,
        debug: bool = False,
        seed: Optional[List[int]] = None,
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if system_instruction is not None:
            raise ValueError("MRCR does not accept a system instruction because it changes the canonical prompt")
        super().__init__(logger=logger)
        self.n_needles = tuple(n_needles) if n_needles is not None else MRCR_NEEDLE_COUNTS
        unknown_needles = set(self.n_needles) - set(MRCR_NEEDLE_COUNTS)
        if unknown_needles:
            raise ValueError(f"unsupported MRCR needle counts: {sorted(unknown_needles)}")
        self.debug = debug
        self.seed = list(seed) if seed is not None else [0, 1234, 1234, 1234]
        self.max_new_tokens = max_tokens
        self.cache_dir = cache_dir if cache_dir is not None else os.environ.get("HF_HUB_CACHE")

    def _load_rows(self) -> Iterable[Dict[str, Any]]:
        dataset = load_dataset(
            DATASET_NAME,
            revision=DATASET_REVISION,
            data_files=list(DATA_FILES),
            streaming=True,
            cache_dir=self.cache_dir,
        )["train"]
        return iter(dataset)

    def _collect_cells(
        self,
        model: LM,
        desired_cells: List[tuple[int, int]],
    ) -> Dict[tuple[int, int], List[SelectedExample]]:
        cells: Dict[tuple[int, int], List[SelectedExample]] = defaultdict(list)
        desired_cell_set = set(desired_cells)
        for source in self._load_rows():
            needles = int(source["n_needles"])
            if needles not in self.n_needles:
                continue
            upper = mrcr_bin(official_token_count(source))
            if upper is None:
                continue
            cell = (upper, needles)
            if cell not in desired_cell_set:
                continue

            example = dict(source)
            messages = parse_prompt(str(example["prompt"]))
            payload = self._prepare_messages(messages, model)
            example["mrcr_bin_upper"] = upper
            cells[cell].append((example, payload))
        return cells

    def _selected_examples(self, model: LM) -> List[SelectedExample]:
        context_length = require_context_length(self._evaluation_max_length, task_name="MRCR")
        eligible_bins = tuple(upper for upper in MRCR_BIN_UPPER_BOUNDS if upper <= context_length)
        desired_cells = [(upper, needles) for upper in eligible_bins for needles in self.n_needles]
        if not desired_cells:
            raise ContextWindowExceededError(
                context_length=context_length,
                required_tokens=MRCR_BIN_UPPER_BOUNDS[0],
            )

        limit = self.evaluation_limit
        if self.debug:
            limit = min(limit, 2) if limit is not None else 2
        cells = self._collect_cells(model, desired_cells)
        missing_cells = [cell for cell in desired_cells if not cells[cell]]
        if missing_cells:
            raise ValueError(f"pinned MRCR dataset has no examples for requested cells: {missing_cells}")
        ordered = _interleave_cells(cells, desired_cells, limit)

        self.logger.info(
            "Selected %d MRCR examples across %d cells",
            len(ordered),
            len({(example["mrcr_bin_upper"], int(example["n_needles"])) for example, _ in ordered}),
        )
        return ordered

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        selected = self._selected_examples(model)
        instances = [
            Instance(
                "generate_until",
                example,
                (
                    payload,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "seed": self.seed,
                    },
                ),
                index,
            )
            for index, (example, payload) in enumerate(selected)
        ]
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None

        examples = [example for example, _ in selected]
        for example, output in zip(examples, outputs):
            score, prefix_hit = score_response(
                output,
                str(example["answer"]),
                str(example["random_string_to_prepend"]),
            )
            example["model_output"] = output
            example["model_answer"] = output
            example["score"] = score
            example["prefix_hit"] = prefix_hit
            example["correct"] = score
        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if results is None:
            return None
        examples = results["examples"]
        if not examples:
            raise ValueError("MRCR cannot aggregate an empty result set")

        metrics: Dict[str, Any] = {
            "num_total": len(examples),
            "mrcr_accuracy": sum(float(example["score"]) for example in examples) / len(examples),
            "prefix_hit_rate": sum(float(example["prefix_hit"]) for example in examples) / len(examples),
            "examples": examples,
        }
        cells: Dict[tuple[int, int], List[float]] = defaultdict(list)
        for example in examples:
            cells[(int(example["mrcr_bin_upper"]), int(example["n_needles"]))].append(float(example["score"]))
        for (upper, needles), scores in sorted(cells.items()):
            metric = f"mrcr_{upper}_{needles}needle"
            metrics[metric] = sum(scores) / len(scores)
            metrics[f"{metric}_count"] = len(scores)
        return metrics

    def _sample_doc(self, example: Dict[str, Any]) -> Dict[str, Any]:
        source = super()._sample_doc(example)
        source.pop("prefix_hit", None)
        return source
