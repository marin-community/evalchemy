# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard the lean (torch-free) evalchemy install.

The base install evaluates a served OpenAI-compatible endpoint and grades the result, so
it must resolve with no torch/vllm/ray and still import every benchmark with an empty
extra. Run this inside a plain ``uv sync`` env (no extras); it is the ``lean-install``
job in .github/workflows/e2e-ci.yaml. A base dependency that grows a heavy transitive
edge, or a lean benchmark that starts importing one, fails here instead of in the
nightly.
"""

import importlib
import importlib.util
import os
import re
import tomllib

# Heavy, accelerator-oriented packages that live in the local-inference extra. None may
# be importable from the base install.
FORBIDDEN_PACKAGES = ("torch", "vllm", "ray")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAT_BENCHMARKS_DIR = os.path.join(REPO_ROOT, "eval", "chat_benchmarks")


def leaked_heavy_packages() -> list[str]:
    """Return the FORBIDDEN_PACKAGES that are importable in the current env."""
    return [pkg for pkg in FORBIDDEN_PACKAGES if importlib.util.find_spec(pkg) is not None]


def lean_benchmarks() -> list[str]:
    """Return benchmark dirs whose normalized optional dependency is empty."""
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        optional_deps = tomllib.load(f)["project"]["optional-dependencies"]

    benchmarks = []
    for name in sorted(os.listdir(CHAT_BENCHMARKS_DIR)):
        if not os.path.isfile(os.path.join(CHAT_BENCHMARKS_DIR, name, "eval_instruct.py")):
            continue
        extra = re.sub(r"[-_.]+", "-", name).lower()
        if extra in optional_deps and not optional_deps[extra]:
            benchmarks.append(name)
    return benchmarks


def import_benchmark(name: str):
    """Import the explicit source registry entry used by TaskManager."""
    from eval.sources import SOURCES
    from eval.contracts.conformance import validate_custom_benchmark_class
    from eval.task import BaseBenchmark

    module_name, class_name = SOURCES[name]
    module = importlib.import_module(module_name)
    validate_custom_benchmark_class(getattr(module, class_name), BaseBenchmark)
    return module


def main() -> None:
    leaked = leaked_heavy_packages()
    if leaked:
        raise RuntimeError(
            f"Lean install leaked heavy dependencies: {', '.join(leaked)}. "
            "A base dependency in pyproject.toml must have grown a heavy transitive edge -- "
            "move it to an extra or constrain it so the torch-free core install stays lean."
        )

    benchmarks = lean_benchmarks()
    for name in benchmarks:
        import_benchmark(name)

    print(
        f"lean install OK: no {', '.join(FORBIDDEN_PACKAGES)}; "
        f"imported {len(benchmarks)} empty-extra benchmarks ({', '.join(benchmarks)})"
    )


if __name__ == "__main__":
    main()
