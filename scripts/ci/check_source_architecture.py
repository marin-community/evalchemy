# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Statically require benchmark sources to leave execution to the shared runtime."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_OPERATIONS = frozenset({
    "generate_responses", "evaluate_responses", "run_benchmark", "compute", "to_samples",
    "ThreadPoolExecutor", "ProcessPoolExecutor", "generate_until", "loglikelihood", "loglikelihood_rolling",
})


def check_source_architecture() -> None:
    """Reject source-owned execution entry points, model calls, and worker pools."""
    violations = []
    for path in sorted((REPO_ROOT / "eval" / "chat_benchmarks").glob("*/eval_instruct.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                operation = node.name
            elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                operation = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            else:
                continue
            if operation in EXECUTION_OPERATIONS:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: source owns {operation}")
    if violations:
        raise SystemExit("\n".join(violations))


if __name__ == "__main__":
    check_source_architecture()
    print("source architecture OK")
