# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard the lean (torch-free) evalchemy install.

The base install evaluates a served OpenAI-compatible endpoint and grades the result, so
it must resolve with no torch/vllm/ray and still import the math/code core benchmarks.
Run this inside a plain ``uv sync`` env (no extras); it is the ``lean-install`` job in
.github/workflows/e2e-ci.yaml. A base dependency that grows a heavy transitive edge, or a
core benchmark that starts importing one, fails here instead of in the nightly.
"""

import importlib.util
import os
import sys

# Heavy, accelerator-oriented packages that live in extras (vllm / benchmarks). None may
# be importable from the base install; each is a top-level import on a benchmark or engine
# path we deliberately keep out of the core.
FORBIDDEN_PACKAGES = ("torch", "vllm", "ray")

# The math/code benchmarks the lean install must support -- these grade against a served
# endpoint with no per-benchmark torch/provider deps, so they must import with no extras.
CORE_BENCHMARKS = (
    "AIME24",
    "AIME25",
    "AMC23",
    "MATH500",
    "HMMT",
    "OlympiadBench",
    "GPQADiamond",
    "LiveCodeBench",
    "MMLUPro",
    "AIW",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAT_BENCHMARKS_DIR = os.path.join(REPO_ROOT, "eval", "chat_benchmarks")


def leaked_heavy_packages() -> list[str]:
    """Return the FORBIDDEN_PACKAGES that are importable in the current env."""
    return [pkg for pkg in FORBIDDEN_PACKAGES if importlib.util.find_spec(pkg) is not None]


def import_benchmark(name: str) -> None:
    """Exec-import a benchmark's eval_instruct.py the way eval/task.py loads it.

    Mirrors TaskManager._load_benchmarks: put the benchmark dir on sys.path and exec the
    module from its file so its sibling imports resolve.
    """
    benchmark_dir = os.path.join(CHAT_BENCHMARKS_DIR, name)
    eval_path = os.path.join(benchmark_dir, "eval_instruct.py")
    sys.path.insert(0, benchmark_dir)
    try:
        spec = importlib.util.spec_from_file_location(f"eval.chat_benchmarks.{name}.eval_instruct", eval_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)


def main() -> None:
    # eval/task.py sets an empty key so benchmarks that read OPENAI_API_KEY at import do
    # not raise on a missing var; mirror that here.
    os.environ.setdefault("OPENAI_API_KEY", "")

    leaked = leaked_heavy_packages()
    if leaked:
        raise RuntimeError(
            f"Lean install leaked heavy dependencies: {', '.join(leaked)}. "
            "A base dependency in pyproject.toml must have grown a heavy transitive edge -- "
            "move it to an extra or constrain it so the torch-free core install stays lean."
        )

    for name in CORE_BENCHMARKS:
        import_benchmark(name)

    print(
        f"lean install OK: no {', '.join(FORBIDDEN_PACKAGES)}; "
        f"imported {len(CORE_BENCHMARKS)} core benchmarks ({', '.join(CORE_BENCHMARKS)})"
    )


if __name__ == "__main__":
    main()
