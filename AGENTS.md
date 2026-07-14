# Evalchemy — agent guide

@.agents/marin-style/AGENTS-core.md

The file above is the shared Marin coding standard (vendored from
[marin-style](https://github.com/marin-community/marin-style); re-vendor with
`marin-style sync`, do not hand-edit `.agents/`). Its companion,
[`.agents/marin-style/TESTING-core.md`](.agents/marin-style/TESTING-core.md), is the
testing policy — read it before writing or reviewing tests. This file adds what is
specific to this repo, and the fork policy below constrains where the standards apply.

Evalchemy is an LLM evaluation harness: it wraps
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with a
larger set of chat/reasoning benchmarks. Marin uses it as the eval half of the
post-training loop — serve a model, evaluate it, gate the scores.

## Install

Evalchemy uses [uv](https://docs.astral.sh/uv/); `uv sync` resolves from the committed
`uv.lock` into `.venv`. Run commands with `uv run ...`.

```bash
make install                                        # uv sync (Python 3.11) + pre-commit hooks
uv sync --extra serve-eval                          # + the serve-and-eval runner / regression gate
uv sync --extra vllm                                # + the local vLLM inference engine
```

The base install is deliberately vLLM-free: evaluating a *served* OpenAI-compatible
endpoint needs no local inference engine, so evalchemy installs on a CPU box. Add
`--extra vllm` only to serve models locally (`--model vllm`).

`marin-serve` is **not** an extra. It cannot co-resolve with evalchemy's dependencies,
so install it as an isolated tool — see the comment in `pyproject.toml`:

```bash
uv tool install --prerelease allow "marin-core>=0.2.0.dev0"
```

## Test

```bash
make test                       # uv run pytest (testpaths = tests/)
uv run pytest tests/e2e -q      # the harness tests: no model, no cluster, seconds
```

`tests/e2e/` covers the runner and the gate against a stdlib HTTP stub and a PTY, and
needs no evalchemy install — CI runs it in an ephemeral env (`uv run --no-project`).

## Lint

```bash
infra/pre-commit.py --all-files --fix    # the required entry point
infra/pre-commit.py --review             # lint-review pass; run before opening a PR
```

This shim runs the pinned `marin-style` checks. Per the fork policy it is scoped to the
Marin-owned directories (see `[tool.marin-style]` in `pyproject.toml`) and runs
`ruff-check` only — no formatter. Formatting stays with upstream's black hook, installed
by `make install` and run on commit (`.pre-commit-config.yaml`); it is not a CI gate.

## Serve-and-eval and the regression gate

Two Marin-owned packages sit on top of evalchemy. Each has a README with the full
interface; the short version:

- **`eval/serve_eval/`** — [README](eval/serve_eval/README.md). Provisions a TPU via
  `marin-serve` (Iris → vLLM) or attaches to an OpenAI-compatible endpoint you already
  have, runs `eval.eval` against it, and prints each task's metrics. A thin orchestrator:
  all of evalchemy's task/scoring/pass@k machinery runs as-is. The `endpoint` provider is
  the hardware-free path.
- **`eval/regression/`** — [README](eval/regression/README.md). Gates a run's
  `results_*.json` against a checked-in spec (`specs/*.json`), or records a new one. The
  gate is a connectivity + coarse-quality smoke check — it asserts the endpoint answered
  the expected number of queries and that each metric clears a wide floor, because scores
  swing run-to-run at small `--limit`. Do not tighten a floor into a two-sided band
  without the variance to justify it.

The runner prints scores; the gate decides pass/fail. Keep that split.

## CI

- **`marin-ci.yaml`** — every PR: `infra/pre-commit.py` over the changed files, and
  `marin-style sync --check` to catch a drifted `.agents/` vendor.
- **`e2e-ci.yaml`** — every PR: the `tests/e2e` harness. Plus two opt-in jobs — an
  endpoint smoke when the `E2E_BASE_URL` repo variable is set, and a `cluster-preflight`
  that checks GCP auth and Iris reachability without provisioning a TPU (label a PR
  `e2e-preflight`, or dispatch it).
- **`e2e-nightly.yaml`** — 07:00 UTC: the real path. Provisions a TPU through
  marin-serve, evaluates, gates against the spec, and tears the slice down in a
  post-step. Keyless GCP auth via workload identity federation
  (`scripts/ci/setup-github-wif.sh` provisions the binding).

The nightly is the only job that touches a cluster. Never make a PR-level job provision
an accelerator.

CI serves Marin's requirements, not upstream's: it lints the Marin-owned code and
exercises the Marin-owned e2e path. It deliberately does **not** run the upstream
benchmark test suite or format-check the upstream tree — that work belongs to
mlfoundations/evalchemy, and policing it here would only block Marin PRs on code the
fork policy forbids us from touching.

## Fork policy

This repo is a fork of [mlfoundations/evalchemy](https://github.com/mlfoundations/evalchemy)
and still tracks upstream. Most of `eval/` is upstream code.

**Marin owns** `eval/serve_eval/`, `eval/regression/`, `tests/e2e/`, `scripts/ci/`,
`infra/`, and the Marin workflows. New Marin work goes here, and the Marin standards
(`infra/pre-commit.py`) apply here.

**Upstream owns** the rest — the benchmark implementations in `eval/chat_benchmarks/`,
`eval/eval.py`, `database/`, `configs/`. Touch it only to fix a real bug or to carry a
change Marin needs, and keep those diffs minimal and surgical:

- **Never reformat upstream code.** No drive-by black/ruff/isort sweeps, no import
  reordering, no docstring rewrites. Every gratuitous line you touch is a merge conflict
  the next upstream pull has to resolve by hand. This is why `[tool.marin-style]` has an
  `include` allowlist rather than linting the whole tree.
- Prefer adding a Marin-owned module over editing an upstream one.
- When you must edit upstream code, say why in the commit message.
