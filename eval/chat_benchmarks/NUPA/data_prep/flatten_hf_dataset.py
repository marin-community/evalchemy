"""Convert the original nested NUPA JSON to row-oriented JSONL and optionally publish it.

Example:
    uv run --extra nupa python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \
        --split test --output /tmp/nupa_test.jsonl \
        --repo-id TODO_ORG/nupa-text-eval
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ijson
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from eval.chat_benchmarks.NUPA.eval_instruct import PUBLISHED_DATASET_NAME, SOURCE_DATASET_NAME, flatten_nupa_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=SOURCE_DATASET_NAME)
    parser.add_argument("--revision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repo-id",
        metavar=PUBLISHED_DATASET_NAME,
        help=(
            "Optional Hugging Face dataset repository to publish. "
            f"The integration placeholder is {PUBLISHED_DATASET_NAME}."
        ),
    )
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--limit-per-task-digit",
        type=int,
        help="Optional deterministic cap applied before flattening each task/digit group.",
    )
    args = parser.parse_args()

    source = args.source_file or Path(
        hf_hub_download(
            repo_id=args.dataset_name,
            filename=f"{args.split}.json",
            repo_type="dataset",
            revision=args.revision,
        )
    )
    count = convert_file(
        source,
        args.output,
        split=args.split,
        limit_per_task_digit=args.limit_per_task_digit,
    )
    print(f"Wrote {count} flattened NUPA records to {args.output}")

    if args.repo_id:
        source_revision = args.revision or HfApi().dataset_info(args.dataset_name).sha
        publish_dataset(
            args.output,
            repo_id=args.repo_id,
            config_name=args.config_name,
            split=args.split,
            private=args.private,
            source_dataset=args.dataset_name,
            source_revision=source_revision,
        )
        print(f"Published https://huggingface.co/datasets/{args.repo_id}")


def convert_file(
    source: Path,
    output: Path,
    *,
    split: str,
    limit_per_task_digit: int | None = None,
) -> int:
    """Stream a nested NUPA JSON file into row-oriented JSONL records."""
    if limit_per_task_digit is not None and limit_per_task_digit <= 0:
        raise ValueError("limit_per_task_digit must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("rb") as source_file, output.open("w", encoding="utf-8") as output_file:
        for task_name, by_digit in ijson.kvitems(source_file, ""):
            row = {task_name: _limit_task(by_digit, limit_per_task_digit)}
            for record in flatten_nupa_row(row, split=split):
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
    return count


def publish_dataset(
    path: Path,
    *,
    repo_id: str,
    config_name: str,
    split: str,
    private: bool,
    source_dataset: str,
    source_revision: str,
) -> None:
    """Upload flattened JSONL plus a provenance-bearing dataset card."""
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        split=split,
        private=private,
        commit_message=f"Publish flattened NUPA {split} split",
    )
    generated_card = Path(
        hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset", force_download=True)
    ).read_text()
    provenance = _provenance(source_dataset, source_revision, config_name, split)
    card = generated_card.split("<!-- nupa-provenance -->", 1)[0].rstrip() + provenance
    HfApi().upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document NUPA source provenance",
    )


def _limit_task(by_digit: Any, limit: int | None) -> dict[str, list[str]]:
    if not isinstance(by_digit, Mapping):
        raise ValueError(f"Expected digit mapping, got {type(by_digit).__name__}")
    limited = {}
    for digit, examples in by_digit.items():
        if not isinstance(examples, list):
            raise ValueError(f"Expected example list for digit {digit}, got {type(examples).__name__}")
        limited[str(digit)] = examples if limit is None else examples[:limit]
    return limited


def _provenance(source_dataset: str, source_revision: str, config_name: str, split: str) -> str:
    return f"""

<!-- nupa-provenance -->

## NUPA text data for Evalchemy

This dataset is a row-oriented conversion of
[`{source_dataset}`](https://huggingface.co/datasets/{source_dataset}) for the
native Evalchemy `NUPA` benchmark. It separates the one-time conversion of the
original nested JSON from model evaluation.

Source revision: `{source_revision}`. Configuration: `{config_name}`. Split:
`{split}`. The source dataset is MIT-licensed; consult its dataset card for the
license terms and original provenance.

Each row contains:

- `id`: stable split, task, digit, and example identifier
- `task_name`: original NUPA task-family and representation key
- `operation`: numeric operation derived from the task key
- `answer_format`: `Integer`, `Float`, `Fraction`, or `ScientificNotation`
- `digit`: original digit group
- `length_bucket`: `S`, `M`, `L`, or `XL`
- `prompt`: model input ending at the source answer delimiter
- `answer`: reference representation used for scoring

Reproduce the conversion from Evalchemy:

```bash
uv run --extra nupa python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \\
  --dataset-name {source_dataset} \\
  --revision {source_revision} \\
  --split {split} \\
  --output /tmp/nupa_{split}.jsonl \\
  --repo-id OWNER/nupa-text-eval
```

Datasets published with `--limit-per-task-digit` are integration fixtures. Do
not use a limited conversion to report benchmark performance.
"""


if __name__ == "__main__":
    main()
