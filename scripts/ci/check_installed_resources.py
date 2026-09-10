# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise packaged benchmark resources from an unrelated working directory."""

from eval.chat_benchmarks.MMLUPro.eval_instruct import generate_cot_prompt


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


if __name__ == "__main__":
    main()
