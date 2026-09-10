# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise packaged benchmark resources from an unrelated working directory."""

from importlib.resources import files
from pathlib import Path

from eval.chat_benchmarks.MMLUPro.eval_instruct import generate_cot_prompt
from eval.contracts.conformance import discover_task_contracts


def main() -> None:
    record = {
        "answer": "A",
        "category": "math",
        "cot_content": "A: Let's think step by step. A",
        "options": ["A", "B"],
        "question": "Which option is first?",
    }
    prompt = generate_cot_prompt([record], record, 1)
    if "Which option is first?" not in prompt:
        raise RuntimeError("installed MMLU-Pro prompt resource produced an invalid prompt")

    benchmark_root = Path(str(files("eval.chat_benchmarks")))
    lm_eval_root = Path(str(files("eval").joinpath("lm_eval_tasks")))
    discover_task_contracts(benchmark_root, lm_eval_root)


if __name__ == "__main__":
    main()
