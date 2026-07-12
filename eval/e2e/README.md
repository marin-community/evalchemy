# Run evalchemy against an Iris/Marin (or any OpenAI) endpoint

- `eval.e2e.run_evals` — serve a model and evaluate it. It provisions a TPU via
  Marin's `marin-serve` (Iris → vLLM), or attaches to an OpenAI-compatible server
  you already have, then runs `python -m eval.eval` against the endpoint and prints
  each task's metrics.
- `eval.e2e.validate` — the CI check. `check` compares a run's `results_*.json`
  against a checked-in baseline and exits non-zero on regression; `record` writes a
  baseline from a run.

## Install

```bash
uv sync --no-dev --python 3.11        # evalchemy base (vLLM-free)

# marin-serve provider only -- an isolated tool, since marin-core can't co-resolve
# with evalchemy's deps:
uv tool install --prerelease allow "marin-core>=0.2.0.dev0"
```

`marin-serve` bundles its working directory as the Iris job workspace, so run it
from a marin checkout via `--marin-workspace <path>`, and keep that checkout free of
large untracked directories — the bundle is `rglob` minus `.gitignore` with a 25 MB
limit ([marin-community/marin#7106](https://github.com/marin-community/marin/issues/7106)).
The `endpoint` provider needs none of this.

## Run

```bash
# Provision a TPU on the marin cluster, eval, print results:
uv run python -m eval.e2e.run_evals --model Qwen/Qwen3-0.6B \
    --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

# Or attach to a server you already have:
uv run python -m eval.e2e.run_evals --provider endpoint --base-url http://localhost:8000/v1

# Gate a run against the baseline (exit 1 on regression), or record a new one:
uv run python -m eval.e2e.validate check  --results eval/e2e/runs/<ts> --baseline eval/e2e/baselines/qwen3-0.6b.json
uv run python -m eval.e2e.validate record --results eval/e2e/runs/<ts> --baseline eval/e2e/baselines/qwen3-0.6b.json
```

`--limit` caps samples per task (default 200; `0` runs the full task; CI uses 20).
Anything after `--` is forwarded to `eval.eval`, e.g. pass@k on a sampled task:

```bash
uv run python -m eval.e2e.run_evals --tasks MATH500 --provider endpoint -- --num_samples 8 --pass_at_k 1,8,32
```

## The gate

gsm8k `strict-match` swings a few samples run-to-run at `--limit 20` even greedy
(vLLM/TPU batching isn't bit-exact), so `compare.py` checks two things that tolerate
that noise: the endpoint answered `expected_samples` queries, and each metric clears
a floor (`record` sets `min = max(0.05, observed − 0.25)`). `MetricThreshold` also
accepts an optional `reference ± tolerance` band for a tighter gate at higher limits.
Pin `provenance.model_revision` in the baseline before committing floors — the Hub
tag is mutable.

Config defaults are in `config.yaml`. CI runs from `.github/workflows/e2e-ci.yaml`
(per-PR) and `e2e-nightly.yaml` (cluster run); the secrets the nightly needs are
documented in its workflow header.
