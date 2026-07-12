# Serve a model and run evalchemy against it

`eval.serve_eval.run` provisions a TPU via Marin's `marin-serve` (Iris → vLLM), or
attaches to an OpenAI-compatible server you already have, then runs `python -m
eval.eval` against the endpoint and prints each task's metrics. It is a thin
orchestrator over `eval.eval` — all of evalchemy's task/scoring/pass@k machinery runs
as-is.

This is the runner; it prints scores. Gating a run against a checked-in threshold spec
is the regression gate's job — see `eval/regression/`.

## Install

```bash
uv sync --no-dev --python 3.11 --extra serve-eval   # evalchemy + the runner (vLLM-free)

# marin-serve provider only -- an isolated tool, since marin-core can't co-resolve
# with evalchemy's deps:
uv tool install --prerelease allow "marin-core>=0.2.0.dev0"
```

`marin-serve` bundles its working directory as the Iris job workspace, so run it from a
marin checkout via `--marin-workspace <path>`, and keep that checkout free of large
untracked directories — the bundle is `rglob` minus `.gitignore` with a 25 MB limit
([marin-community/marin#7106](https://github.com/marin-community/marin/issues/7106)).
The `endpoint` provider needs none of this.

## Run

```bash
# Provision a TPU on the marin cluster, eval, print results:
uv run python -m eval.serve_eval.run --model Qwen/Qwen3-0.6B \
    --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

# Or attach to a server you already have:
uv run python -m eval.serve_eval.run --provider endpoint --base-url http://localhost:8000/v1
```

`--limit` caps samples per task (default 200; `0` runs the full task; CI uses 20).
Anything after `--` is forwarded to `eval.eval`, e.g. pass@k on a sampled task:

```bash
uv run python -m eval.serve_eval.run --tasks MATH500 --provider endpoint -- --num_samples 8 --pass_at_k 1,8,32
```

Config defaults are in `configs/qwen-tiny.yaml` (loaded with pydantic-settings; CLI
flags and `E2E_*` env vars override it). Two providers:

- `endpoint` — attach to a running `/v1` server (the verified, hardware-free path).
- `marin-serve` — provisions the TPU itself in `--access link` (mint) mode, which
  returns a PUBLIC capability URL with a scoped token in the path, so no auth header
  or SSH tunnel is needed. `--access private` keeps the proxy cluster-only.

Chat models use `local-chat-completions` + a bare `--apply_chat_template` flag +
`tokenizer_backend=huggingface,tokenized_requests=False` (the served model tokenizes;
lm-eval keeps a HF tokenizer only for length bookkeeping).

To gate a run's scores against a checked-in spec, hand the output dir to the gate:

```bash
uv run python -m eval.regression.validate check --results <output-dir> \
    --spec eval/regression/specs/qwen3-0.6b.json
```
