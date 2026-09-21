"""Materialize Evalchemy's pinned NUPA sample as row-oriented JSONL.

Example:
    uv run --extra nupa python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \
        --output /tmp/nupa_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from eval.chat_benchmarks.NUPA.eval_instruct import (
    DEFAULT_NUM_EACH,
    DEFAULT_RANDOM_SEED,
    SOURCE_DATASET_NAME,
    SOURCE_DATASET_REVISION,
    iter_nupa_source_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=SOURCE_DATASET_NAME)
    parser.add_argument("--revision", default=SOURCE_DATASET_REVISION)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-each", type=int, default=DEFAULT_NUM_EACH)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
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
        num_each=args.num_each,
        random_seed=args.random_seed,
    )
    print(f"Wrote {count} flattened NUPA records to {args.output}")


def convert_file(
    source: Path,
    output: Path,
    *,
    split: str,
    num_each: int = DEFAULT_NUM_EACH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    """Write the deterministic NUPA evaluation sample to JSONL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as output_file:
        for record in iter_nupa_source_records(
            source,
            split=split,
            num_each=num_each,
            random_seed=random_seed,
        ):
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    main()
