"""Build the deterministic 39-row AMO-Bench parser subset from pinned parquet."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


SOURCE_SHA256 = "eb08f5239c0cc092e13552c31e505a7fdb8e8140aa8e227beedd62fc442cde58"
EXPECTED_TYPE_COUNTS = {"number": 34, "set": 3, "variable": 2, "description": 11}
VARIABLE_CASES = {
    5: [f"n={value}" for value in range(1, 21)],
    37: [f"a={value},b={value + 1},c={value + 2}" for value in range(2, 19)],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, output: Path) -> None:
    actual_hash = sha256(source)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Unexpected AMO-Bench source hash: {actual_hash}")

    rows = pd.read_parquet(source).to_dict(orient="records")
    type_counts = Counter(str(row["answer_type"]) for row in rows)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"Unexpected AMO-Bench type counts: {dict(type_counts)}")

    parser_rows = []
    for row in rows:
        if row["answer_type"] == "description":
            continue
        normalized = {
            "question_id": int(row["question_id"]),
            "problem": str(row["prompt"]),
            "answer": str(row["answer"]),
            "answer_type": str(row["answer_type"]),
        }
        if normalized["answer_type"] == "variable":
            normalized["verification_cases"] = VARIABLE_CASES[normalized["question_id"]]
        parser_rows.append(normalized)

    if len(parser_rows) != 39:
        raise ValueError(f"Expected 39 parser-gradeable rows, found {len(parser_rows)}")
    if {row["question_id"] for row in parser_rows if row["answer_type"] == "variable"} != set(VARIABLE_CASES):
        raise ValueError("AMO-Bench variable problem IDs changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in parser_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Pinned AMO-Bench test parquet")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/chat_benchmarks/AMOBenchParser/data/amo_bench_parser.jsonl"),
    )
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
