"""Reproduce the fixed optimization subset from the policy NUPA5K manifest."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval/chat_benchmarks/NUPA5K-Loose/data/nupa5k_manifest.jsonl"
OUTPUT = ROOT / "eval/chat_benchmarks/NUPA200-Loose/data/nupa200_manifest.jsonl"


def main() -> None:
    groups = defaultdict(list)
    for line in SOURCE.read_text().splitlines():
        record = json.loads(line)
        groups[record["task_name"]].append(record)
    names = sorted(groups, key=lambda name: hashlib.sha256(("nupa200-v1:" + name).encode()).hexdigest())
    selected = []
    for index, name in enumerate(names):
        records = sorted(groups[name], key=lambda record: (record["digit"], record["sha256"]))
        count = 4 + (index < 24)
        selected.extend(records[((2 * offset + 1) * len(records)) // (2 * count)] for offset in range(count))
    selected.sort(key=lambda record: (record["task_name"], record["digit"], record["sha256"]))
    OUTPUT.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in selected))


if __name__ == "__main__":
    main()
