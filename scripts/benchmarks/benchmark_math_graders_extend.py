"""Time the self-contained MATH graders in eval/graders on the 1000-problem benchmark.
Usage:
    uv run python scripts/benchmarks/benchmark_math_graders_extend.py
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import hendrycks_math, minerva_math  # noqa: E402

BENCHMARK = REPO / "tests/graders/data/math_extended_1000_grader_benchmark.jsonl"

GRADERS = [
    (
        "hendrycks_math",
        hendrycks_math,
        "hendrycks_solution",
        "hendrycks_reference_grade",
        hendrycks_math.extract_answer,
    ),
    (
        "minerva_math",
        minerva_math,
        "minerva_solution",
        "minerva_reference_grade",
        minerva_math.extract_answer,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timed runs after the cold run")
    args = parser.parse_args()

    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    print(f"{len(records)} problems from {BENCHMARK.name}\n")

    # Importing sympy and building its ANTLR parser tables is a one-time cost
    # the reference pays too, so charge it before timing any grading.
    start = time.perf_counter()
    minerva_math.is_equiv("\\sqrt{2}", "\\sqrt{2}")
    print(f"one-time sympy + ANTLR init: {(time.perf_counter() - start) * 1e3:.1f} ms\n")

    for name, grader, solution_field, reference_field, extract_answer in GRADERS:
        if grader is minerva_math:
            minerva_math._sympy_parses.cache_clear()

        start = time.perf_counter()
        grades = [grader.grade(r["problem"], r[solution_field], r["reference_answer"]) for r in records]
        cold = time.perf_counter() - start

        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for record in records:
                grader.grade(record["problem"], record[solution_field], record["reference_answer"])
            runs.append(time.perf_counter() - start)

        expected = [r[reference_field] for r in records]
        disagreements = [i for i, (got, want) in enumerate(zip(grades, expected, strict=True)) if got != want]

        print(f"{name}")
        print(f"  score          {sum(grades):.0f}/{len(records)} (harness: {sum(expected):.0f})")
        print(f"  agreement      {len(records) - len(disagreements)}/{len(records)} with lm-eval-harness")
        print(f"  cold cache     {cold * 1e3:7.2f} ms  ({cold / len(records) * 1e3:6.3f} ms/problem)")
        print(
            f"  warm (median)  {statistics.median(runs) * 1e3:7.2f} ms  "
            f"({statistics.median(runs) / len(records) * 1e3:6.3f} ms/problem)"
        )
        for index in disagreements:
            record = records[index]
            extracted = extract_answer(record[solution_field])
            print(
                f"  DISAGREE #{index} kind={record['kind']} "
                f"extracted={extracted!r} reference_answer={record['reference_answer']!r}"
            )
        print()


if __name__ == "__main__":
    main()
