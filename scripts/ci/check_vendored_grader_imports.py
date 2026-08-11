# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Statically audit vendored grader packages for undeclared module-load imports.

A *standalone vendored grader* is a self-contained package shipped inside a benchmark
dir (e.g. ``eval/chat_benchmarks/HumanEvalPlus/human_eval_plus/``) that the benchmark's
``eval_instruct.py`` imports as a top-level package (``from human_eval_plus.evaluation
import ...``). ``eval/task.py`` execs ``eval_instruct.py`` with the benchmark dir on
``sys.path``, so importing the grader runs its module-body imports immediately. If any of
those names a third-party package that is neither stdlib nor declared in the base install
or the benchmark's extra, the grader raises ``ModuleNotFoundError`` at import time --
exactly the ``human_eval_plus`` / ``mbpp_plus`` failure this guard exists to catch.

``check_benchmark_extras.py`` proves a benchmark *imports* in an isolated env, but a
grader dep that is only transitively reachable slips through: ``regex`` is imported at
module load by the HumanEval(+)/MBPP+ graders and the HMMT ``matharena`` grader, and it
happens to ride in on ``tiktoken``/``transformers``/``nltk``, so the isolated env still
resolves it. A resolver change (or a stripped install) drops that transitive edge and the
import breaks. This check is STATIC (``ast`` + ``tomllib`` -- no install, no network): it
traces the module-body import chain each grader actually triggers and asserts every
third-party import is declared in ``pyproject.toml``, failing on the undeclared one
instead of on the transitive coincidence.

Vendored *frameworks* (MTBench's ``fastchat``, MixEval's ``mix_eval``) are deliberately
out of scope: their import closures are large and full of try/except optional imports, so
static tracing is noisy; they stay covered by ``check_benchmark_extras.py`` and their
declared extras. The table below lists only the standalone graders; extend it (and the
``TRUSTED_TRANSITIVE`` set, with a justification) when adding one.

Usage:
  uv run --no-project --python 3.12 python scripts/ci/check_vendored_grader_imports.py
"""

import ast
import os
import re
import sys
import tomllib

# These siblings (also stdlib-only) are the canonical sources for the repo/benchmark-dir
# roots and the PEP-685 extra-name normalizer; reuse them rather than re-deriving the
# paths or the regex a third time.
from check_lean_install import CHAT_BENCHMARKS_DIR, REPO_ROOT
from check_benchmark_extras import normalize_extra

# Standalone vendored grader packages: benchmark dir -> the top-level import name of the
# vendored package. Each is a self-contained scorer (pass@k / math answer extraction)
# whose module-body imports run as soon as eval/task.py execs eval_instruct.py.
VENDORED_GRADERS: dict[str, str] = {
    "HumanEval": "human_eval",
    "HumanEvalPlus": "human_eval_plus",
    "MBPP": "human_eval",
    "MBPPPlus": "mbpp_plus",
    "MultiPLE": "multiple",
    "HMMT": "matharena",
}

# Third-party modules treated as guaranteed-available from the base install even though
# they are not directly declared in [project.dependencies]. Kept tiny and reviewed: every
# entry must trace to a base dep that reliably pulls it, and a grader relying on anything
# less universal MUST declare it instead. Add a one-line justification per entry.
TRUSTED_TRANSITIVE: set[str] = {
    # datasets + transformers + lm-eval (all core) depend on tqdm; it is present in every
    # base install and the repo declares it nowhere by convention.
    "tqdm",
    # lm-eval[math] (core) depends on sympy + antlr4 for the boxed-answer math grader; the
    # matharena grader (HMMT) imports sympy at module load.
    "sympy",
}

# Top-level packages that are part of this project itself (always importable from a
# source checkout or the installed wheel) and so need no third-party declaration.
SELF_PACKAGES: set[str] = {"eval"}

# PyPI distribution name -> importable top-level module name, for the cases where the
# normalized name (dashes->underscores, lowercased) does not match the import. Everything
# not here is assumed to import under ``name.replace("-", "_").lower()``.
DIST_TO_MODULE: dict[str, str] = {
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
    "antlr4-python3-runtime": "antlr4",
    "bespokelabs-curator": "curator",
    "tree-sitter": "tree_sitter",
    "tree-sitter-python": "tree_sitter_python",
    "google-generativeai": "google",
}


def normalize_dep_name(spec: str) -> str:
    """Importable module name for a PEP-508 dependency spec (best-effort)."""
    n = re.split(r"[\[<>=!~ ;@]", spec, maxsplit=1)[0].strip().lower()
    return DIST_TO_MODULE.get(n, n.replace("-", "_"))


def declared_modules() -> tuple[set[str], dict[str, set[str]]]:
    """Return (base module names, {extra name -> module names declared by that extra})."""
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        pyproject = tomllib.load(f)
    base = {normalize_dep_name(d) for d in pyproject["project"]["dependencies"]}
    extras = {
        name: {normalize_dep_name(d) for d in specs}
        for name, specs in pyproject["project"]["optional-dependencies"].items()
    }
    return base, extras





def _module_body_imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Top-level-body import statements (descend into if/try/with, not def/class).

    Imports inside ``if``/``try``/``with`` at module scope run at import time (guarded or
    not), so they are module-load; imports inside function/class bodies are lazy and are
    NOT collected.
    """
    out: list[ast.Import | ast.ImportFrom] = []

    def visit_body(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(node)
            elif isinstance(node, ast.Try):
                visit_body(node.body)
                visit_body(node.orelse)
                visit_body(node.finalbody)
            elif isinstance(node, ast.If):
                visit_body(node.body)
                visit_body(node.orelse)
            elif isinstance(node, ast.With):
                visit_body(node.body)

    visit_body(tree.body)
    return out


def _internal_submodule(node: ast.Import | ast.ImportFrom, pkg: str) -> str | None:
    """Dotted submodule path WITHIN the vendored package, or None if the import is external.

    Covers ``from .data import x`` (relative) and ``from <pkg>.utils import x`` (absolute
    into the package itself). ``ast.Import`` nodes are always external (top-level names).
    """
    if not isinstance(node, ast.ImportFrom):
        return None
    if node.level and node.level > 0:
        # Relative import: only same-package (level 1 from a top-level module). ``from .``
        # with module=None imports names directly; those names are themselves submodules
        # only when they resolve to a file, handled by the caller via _submodule_file.
        return node.module  # may be None for ``from . import x``; caller walks names
    if node.module and (node.module == pkg or node.module.startswith(pkg + ".")):
        return node.module[len(pkg) + 1 :] if node.module != pkg else ""
    return None


def _submodule_file(pkg_dir: str, dotted: str) -> str | None:
    """On-disk .py for a dotted submodule path inside the package, or None."""
    if not dotted:
        return os.path.join(pkg_dir, "__init__.py")
    rel = dotted.replace(".", os.sep)
    py = os.path.join(pkg_dir, rel + ".py")
    if os.path.isfile(py):
        return py
    init = os.path.join(pkg_dir, rel, "__init__.py")
    if os.path.isfile(init):
        return init
    return None


def eval_instruct_import_roots(benchmark_dir: str, pkg: str) -> set[str]:
    """Dotted submodule paths of ``pkg`` that ``eval_instruct.py`` imports at module load.

    These are the entry points into the vendored package -- the modules whose module-body
    code runs first. The package ``__init__`` is always added by the tracer as a root.
    """
    eval_path = os.path.join(CHAT_BENCHMARKS_DIR, benchmark_dir, "eval_instruct.py")
    roots: set[str] = set()
    # An unreadable or syntactically invalid eval_instruct.py is a real failure, not a
    # reason to trace nothing and silently report OK -- surface it to the caller.
    source = open(eval_path).read()
    tree = ast.parse(source, filename=eval_path)
    for node in _module_body_imports(tree):
        sub = _internal_submodule(node, pkg)
        if sub is None:
            continue
        if sub == "":
            # ``from <pkg> import a, b`` -- a/b may be submodules; best effort: add as-is.
            roots.update(n.name.split(".")[0] for n in node.names)
        else:
            roots.add(sub)
    return roots


def trace_grader_imports(benchmark_dir: str, pkg: str) -> tuple[set[str], set[str]]:
    """Trace a grader's module-load import chain.

    Returns (external_third_party_modules, unresolved_internal_submodules). The external
    set is what the declaration check asserts against; the unresolved set is reported for
    diagnostics only (a missing internal submodule usually means a data-only module).
    """
    pkg_dir = os.path.join(CHAT_BENCHMARKS_DIR, benchmark_dir, pkg)
    external: set[str] = set()
    unresolved: set[str] = set()
    seen_files: set[str] = set()

    # Roots: the package __init__ plus every submodule eval_instruct pulls in.
    roots = {""} | eval_instruct_import_roots(benchmark_dir, pkg)  # "" -> __init__.py
    queue: list[str] = sorted(roots)
    queued: set[str] = set(queue)

    while queue:
        dotted = queue.pop()
        path = _submodule_file(pkg_dir, dotted)
        if path is None or path in seen_files:
            if path is None and dotted:
                unresolved.add(dotted)
            continue
        seen_files.add(path)
        # A grader submodule that cannot be read or parsed would leave its imports
        # unaudited and produce a false OK -- let the error propagate instead.
        tree = ast.parse(open(path).read(), filename=path)

        for node in _module_body_imports(tree):
            sub = _internal_submodule(node, pkg)
            if sub is not None:
                # Internal: recurse into the submodule (and into ``from . import x`` names).
                targets = [sub] if sub else [n.name.split(".")[0] for n in node.names]
                for t in targets:
                    if t and t not in queued and _submodule_file(pkg_dir, t):
                        queued.add(t)
                        queue.append(t)
                continue
            # External: record the top-level module name (skip stdlib and this project's
            # own packages -- they are never a missing-dependency failure).
            if isinstance(node, ast.Import):
                for n in node.names:
                    top = n.name.split(".")[0]
                    if top not in sys.stdlib_module_names and top not in SELF_PACKAGES:
                        external.add(top)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top not in sys.stdlib_module_names and top not in SELF_PACKAGES:
                    external.add(top)

    return external, unresolved


def main() -> None:
    base, extras = declared_modules()
    failures: list[str] = []

    print("=== vendored grader module-load import audit ===")
    for benchmark_dir in sorted(VENDORED_GRADERS):
        pkg = VENDORED_GRADERS[benchmark_dir]
        extra = normalize_extra(benchmark_dir)
        allowed = base | extras.get(extra, set()) | TRUSTED_TRANSITIVE
        external, unresolved = trace_grader_imports(benchmark_dir, pkg)
        undeclared = sorted(m for m in external if m not in allowed)
        status = "OK" if not undeclared else "FAIL"
        print(f"  [{status}] {benchmark_dir} [{extra}] -> {pkg}")
        print(f"        module-load third-party imports: {sorted(external) or '-'}")
        if unresolved:
            print(f"        (non-package internal refs, ignored: {sorted(unresolved)})")
        for m in undeclared:
            failures.append(
                f"{benchmark_dir}: grader `{pkg}` imports `{m}` at module load, but `{m}` is "
                f"not declared in the base install, the `{extra}` extra, or TRUSTED_TRANSITIVE. "
                "Add it to [project.dependencies] (if a lean benchmark needs it) or the extra."
            )

    print()
    if failures:
        print("FAIL: undeclared module-load grader imports:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"OK: {len(VENDORED_GRADERS)} vendored grader(s) declare every module-load import.")


if __name__ == "__main__":
    main()
