"""Generate the GSM8KPerturbed static datasets.

Sources the canonical openai/gsm8k dataset (MIT): the test split for
questions and gold answers, the train split for history exchanges. Writes

    data/gsm8k_clean.jsonl      the unperturbed reference
    data/gsm8k_perturbed.jsonl  one perturbed instance per (item, condition)

Both files start with a meta line recording the exact command and parameters
that produced them; the benchmark loader skips it. Regenerate (only when the
design changes) with:

    uv run --with datasets --with cmudict python -m eval.chat_benchmarks.GSM8KPerturbed.data_prep.generate

Instance fields:
    id                    "gsm8k-test-#<index>::<condition>"
    condition             one of perturbations.CONDITIONS
    seed                  per-instance transform seed
    original_question     verbatim openai/gsm8k test question
    question              perturbed text (== original for history conditions)
    answer                gold final answer (the value after "#### ")
    changed               question != original_question
    number_words_affected spoken-number homophone swap occurred
    history_train_indices openai/gsm8k train indices for provenance
                          (history conditions only, else null)
    history_exchanges     the exact prepended Q/A pairs, embedded so the
                          benchmark runs fully offline

Design: marin-community/marin#7776 (part of marin-community/marin#7090).
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from eval.chat_benchmarks.GSM8KPerturbed.src.perturbations import (
    CONDITIONS,
    HISTORY_CONDITIONS,
    assign_conditions,
    history_indices,
    perturb_question,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SEED = 20260731


def gold_answer(solution: str) -> str:
    """openai/gsm8k answers end with '#### <final answer>'."""
    marker = "#### "
    assert marker in solution, solution
    return solution.rsplit(marker, 1)[1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--perturbations-per-item", type=int, default=1)
    args = parser.parse_args()

    meta = {
        "meta": {
            "generator": "eval/chat_benchmarks/GSM8KPerturbed/data_prep/generate.py",
            "command": "uv run --with datasets --with cmudict python -m "
            "eval.chat_benchmarks.GSM8KPerturbed.data_prep.generate "
            f"--seed {args.seed} --perturbations-per-item {args.perturbations_per_item}",
            "seed": args.seed,
            "perturbations_per_item": args.perturbations_per_item,
            "source": "openai/gsm8k (main): test split for items, train split for history exchanges",
            "conditions": CONDITIONS,
        }
    }

    test = load_dataset("openai/gsm8k", "main", split="test")
    train = load_dataset("openai/gsm8k", "main", split="train")
    assigned = assign_conditions(len(test), args.seed, args.perturbations_per_item)

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "gsm8k_clean.jsonl", "w") as f:
        f.write(json.dumps(meta) + "\n")
        for ind, item in enumerate(test):
            f.write(json.dumps(
                {"id": f"gsm8k-test-#{ind}", "question": item["question"], "answer": gold_answer(item["answer"])},
                ensure_ascii=False) + "\n")

    n_lines = 0
    with open(DATA_DIR / "gsm8k_perturbed.jsonl", "w") as f:
        f.write(json.dumps(meta) + "\n")
        for ind, (item, conditions) in enumerate(zip(test, assigned)):
            for condition in conditions:
                record = {
                    "id": f"gsm8k-test-#{ind}::{condition}",
                    "condition": condition,
                    "original_question": item["question"],
                    "answer": gold_answer(item["answer"]),
                }
                if condition in HISTORY_CONDITIONS:
                    idxs = history_indices(len(train), ind, HISTORY_CONDITIONS[condition], args.seed)
                    record.update({
                        "seed": args.seed,
                        "question": item["question"],
                        "changed": True,
                        "number_words_affected": False,
                        "history_train_indices": idxs,
                        "history_exchanges": [
                            {"question": train[i]["question"], "answer": train[i]["answer"]} for i in idxs
                        ],
                    })
                else:
                    p = perturb_question(item["question"], condition, args.seed + ind)
                    record.update({
                        "seed": p.seed,
                        "question": p.text,
                        "changed": p.changed,
                        "number_words_affected": p.number_words_affected,
                        "history_train_indices": None,
                        "history_exchanges": None,
                    })
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_lines += 1
    print(f"wrote {DATA_DIR}/gsm8k_clean.jsonl ({len(test)} items) and gsm8k_perturbed.jsonl ({n_lines} instances)")


if __name__ == "__main__":
    sys.exit(main())
