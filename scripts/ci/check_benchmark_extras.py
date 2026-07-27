# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify each per-benchmark extra imports its benchmark in ISOLATION.

pyproject.toml declares one optional-dependency per eval/chat_benchmarks dir (the PEP-685
normalized dir name). This guards the contract that extra ``x`` carries exactly the
module-level deps benchmark dir X needs beyond the lean base: for each non-empty extra it
syncs a throwaway env with ONLY the base + that one extra, then exec-imports the
benchmark's eval_instruct.py the way eval/task.py does. It also asserts that the resolved
environment has no torch, which belongs to the local-inference extra. A missing or
mis-scoped dependency fails here, in CI, instead of at eval time.

Empty extras (the benchmark rides entirely on the lean base) are covered by
check_lean_install.py, so they are skipped unless ``--all`` is passed.

Modes:
  (outer, default)  Sync one venv per extra under --venv-root, reuse uv's download cache,
                    and dispatch the import into each.
  --import-one NAME Inner mode, run by the per-extra venv's own python: exec-import that
                    one benchmark and exit nonzero on any import error.

Usage:
  uv run python scripts/ci/check_benchmark_extras.py            # all non-empty extras
  uv run python scripts/ci/check_benchmark_extras.py --all      # + the empty ones
  uv run python scripts/ci/check_benchmark_extras.py MTBench IFEval   # just these dirs
"""

import argparse
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass

from check_lean_install import CHAT_BENCHMARKS_DIR, REPO_ROOT, import_benchmark

PYTHON_VERSION = "3.11"

# eval/task.py sets an empty OPENAI_API_KEY so benchmarks that read it at import do not hit
# a NoneType; HLE's judge client rejects an EMPTY key, so use a non-empty dummy here. It
# never leaves the import (no request is made), it only has to be truthy.
HLE_DUMMY_OPENAI_KEY = "ci-dummy"

# One venv per extra lives here across runs so a rerun reuses the resolved env (uv sync is
# idempotent); overridable with --venv-root.
DEFAULT_VENV_ROOT = os.path.join(REPO_ROOT, ".benchmark-extra-venvs")


@dataclass(frozen=True)
class BenchmarkExtra:
    """A benchmark dir paired with the extra that makes it importable."""

    dir_name: str
    extra: str
    is_empty: bool


def normalize_extra(dir_name: str) -> str:
    """PEP-685 extra name for a benchmark dir: lowercase, runs of -_. collapse to one -."""
    return re.sub(r"[-_.]+", "-", dir_name).lower()


def discover_benchmark_extras() -> list[BenchmarkExtra]:
    """Pair every eval/chat_benchmarks dir (with an eval_instruct.py) to its declared extra.

    Reads [project.optional-dependencies] from pyproject.toml as the source of truth for
    which extras exist and which are empty, and asserts every benchmark dir has one -- a
    dir without a matching extra breaks the ``evalchemy[<any benchmark>]`` contract.
    """
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        optional_deps = tomllib.load(f)["project"]["optional-dependencies"]

    extras: list[BenchmarkExtra] = []
    for dir_name in sorted(os.listdir(CHAT_BENCHMARKS_DIR)):
        if not os.path.exists(os.path.join(CHAT_BENCHMARKS_DIR, dir_name, "eval_instruct.py")):
            continue
        extra = normalize_extra(dir_name)
        if extra not in optional_deps:
            raise ValueError(
                f"Benchmark dir {dir_name!r} has no extra {extra!r} in pyproject.toml -- "
                "every benchmark dir must declare an extra (empty if the base covers it) so "
                "evalchemy[<benchmark>] is uniformly valid."
            )
        extras.append(BenchmarkExtra(dir_name, extra, is_empty=not optional_deps[extra]))
    return extras


def sync_extra(extra: str, venv_dir: str) -> None:
    """Sync base + only this extra into venv_dir, against the committed lock (--frozen)."""
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": venv_dir}
    subprocess.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--python",
            PYTHON_VERSION,
            "--extra",
            extra,
            "--reinstall-package",
            "evalchemy",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def import_in_venv(venv_dir: str, dir_name: str) -> subprocess.CompletedProcess:
    """Exec-import one benchmark using the per-extra venv's own python (inner mode)."""
    return subprocess.run(
        [os.path.join(venv_dir, "bin", "python"), os.path.abspath(__file__), "--import-one", dir_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def check_one(target: BenchmarkExtra, venv_root: str) -> bool:
    """Sync the isolated env for one extra and import its benchmark; return True on success."""
    venv_dir = os.path.join(venv_root, f"venv-{target.extra}")
    print(f"\n=== {target.dir_name} [{target.extra}] ===", flush=True)
    sync_extra(target.extra, venv_dir)
    result = import_in_venv(venv_dir, target.dir_name)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


def run_checks(targets: list[BenchmarkExtra], venv_root: str) -> None:
    os.makedirs(venv_root, exist_ok=True)
    results = {t.dir_name: check_one(t, venv_root) for t in targets}

    print("\n==== benchmark extra isolation results ====")
    for dir_name, ok in sorted(results.items()):
        print(f"  {'PASS' if ok else 'FAIL'}  {dir_name}")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        raise SystemExit(f"{len(failed)} extra(s) failed to import in isolation: {', '.join(sorted(failed))}")
    print(f"\nOK: {len(targets)} benchmark extra(s) import in isolation.")


def select_targets(args: argparse.Namespace, all_extras: list[BenchmarkExtra]) -> list[BenchmarkExtra]:
    if args.benchmarks:
        by_dir = {e.dir_name: e for e in all_extras}
        unknown = [name for name in args.benchmarks if name not in by_dir]
        if unknown:
            raise ValueError(f"Unknown benchmark dir(s): {', '.join(unknown)}")
        return [by_dir[name] for name in args.benchmarks]
    if args.all:
        return all_extras
    return [e for e in all_extras if not e.is_empty]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", help="Specific benchmark dir names to check (default: all non-empty).")
    parser.add_argument("--all", action="store_true", help="Include empty extras (equivalent to the lean check).")
    parser.add_argument("--venv-root", default=DEFAULT_VENV_ROOT, help="Where the per-extra venvs live.")
    parser.add_argument("--import-one", metavar="DIR", help="Inner mode: exec-import one benchmark and exit.")
    args = parser.parse_args()

    if args.import_one:
        # Inner mode runs under a per-extra venv's python. HLE's module load needs a
        # truthy OPENAI_API_KEY; set the dummy for every benchmark (harmless -- no request).
        os.environ["OPENAI_API_KEY"] = HLE_DUMMY_OPENAI_KEY
        # Mirror production: eval/task.py (the loader) is already in sys.modules when it
        # execs a benchmark, so `from eval.task import BaseBenchmark` resolves from the
        # cache. Without this, a benchmark dir with a sibling eval.py (RepoBench) shadows
        # the installed `eval` package via the sys.path[0] insert.
        importlib.import_module("eval.task")
        if importlib.util.find_spec("torch") is not None:
            raise RuntimeError(
                f"Benchmark extra for {args.import_one} installed torch; endpoint/grading extras must stay torch-free."
            )
        module = import_benchmark(args.import_one)
        if args.import_one == "BigCodeBench":
            module.cleanup_resources()
        elif args.import_one == "LiveBench":
            conversation = module.get_conversation_template("gpt-4o-mini-2024-07-18")
            if conversation.name != "chatgpt":
                raise RuntimeError("LiveBench's torch-free model adapter selected the wrong conversation template.")
            common = importlib.import_module("livebench.common")
            if common.model_display_name("gpt-4o-mini") != "gpt-4o-mini-2024-07-18":
                raise RuntimeError("LiveBench's torch-free model alias lookup changed.")
            model_adapter = importlib.import_module("livebench.model.model_adapter")
            try:
                model_adapter.load_model("local-model")
            except ModuleNotFoundError as exc:
                if "evalchemy[vllm]" not in str(exc):
                    raise
            else:
                raise RuntimeError("LiveBench local model loading did not require the local-inference extra.")
        elif args.import_one == "MTBench":
            api_models = importlib.import_module("fastchat.model.api_models")
            conversation = api_models.get_api_conversation_template("gpt-4o-mini-2024-07-18")
            if conversation.name != "gpt-mini":
                raise RuntimeError("MTBench's torch-free judge conversation template changed.")
            importlib.import_module("fastchat.llm_judge.gen_api_answer")
        elif args.import_one == "MixEval":
            common_utils = importlib.import_module("mix_eval.utils.common_utils")
            common_utils.set_seed(123)
            first = common_utils.random.random()
            common_utils.set_seed(123)
            if common_utils.random.random() != first:
                raise RuntimeError("MixEval's torch-free grader seeding is not deterministic.")
        elif args.import_one == "alpaca_eval":
            with module._no_grad():
                pass
        print(f"imported {args.import_one} OK")
        return

    run_checks(select_targets(args, discover_benchmark_extras()), args.venv_root)


if __name__ == "__main__":
    main()
