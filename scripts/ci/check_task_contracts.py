# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Enumerate both task registries and require the shared lifecycle contract."""

from pathlib import Path

from eval.contracts.conformance import discover_task_contracts
from eval.contracts.task_outcome import TaskRoute

REPO_ROOT = Path(__file__).parents[2]
LM_EVAL_INCLUDE_DIR = REPO_ROOT / "eval" / "lm_eval_tasks"


def main() -> None:
    custom_root = REPO_ROOT / "eval" / "chat_benchmarks"
    contracts = discover_task_contracts(custom_root, LM_EVAL_INCLUDE_DIR)
    custom_count = sum(contract.route is TaskRoute.CUSTOM for contract in contracts)
    lm_eval_count = len(contracts) - custom_count
    print(
        "task contracts OK: "
        f"{custom_count} custom, {lm_eval_count} lm-eval, {len(contracts)} total"
    )


if __name__ == "__main__":
    main()
