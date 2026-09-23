# Serve a model and run evalchemy against it

`eval.serve_eval.run` provisions a TPU via Marin's `marin-serve` (Iris → vLLM), or
attaches to an OpenAI-compatible server you already have, then runs `python -m
eval.eval` against the endpoint and prints each task's metrics. It is a thin
orchestrator over `eval.eval` — all of evalchemy's task/scoring/pass@k machinery runs
as-is.

This is the runner; it prints scores. Gating a run against a checked-in threshold spec
is the regression gate's job — see `eval/regression/`.

The saved `results_*.json` includes one `task_outcomes` entry per requested task.
Each entry records its status, coverage, classified task failure (if any),
`failure_counts` for endpoint failures, and `completion_response_summary` for chat
responses. A failed task is retained in the package without a score; a malformed
response within a scored task remains an empty sample with `failure_category` in
its sample record. Evalchemy completes the other tasks and leaves the acceptance
decision to the caller or regression gate.

## Install

```bash
uv sync --no-dev --python 3.12 --extra serve-eval   # evalchemy + the runner (vLLM-free)

# marin-serve provider only -- an isolated tool, since marin-core can't co-resolve
# with evalchemy's deps:
uv tool install --prerelease allow "marin-core>=0.2.125.dev35848091506"
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
    --tpu v6e-4,v5litepod-4,v5p-8 --marin-workspace /path/to/marin

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
- `marin-serve` — provisions the TPU itself and returns a capability URL with a
  scoped token in the path, so no auth header or SSH tunnel is needed.

Chat models use `local-chat-completions` + a bare `--apply_chat_template` flag +
`tokenizer_backend=none,tokenized_requests=False`. The endpoint applies its chat
template, so chat evaluation does not load the checkpoint's tokenizer in the client.
With no client tokenizer, Evalchemy logs that prompt-length preflight was skipped
and sends the requested generation cap unchanged. The endpoint remains responsible
for enforcing its context window on those requests.
Plain completions use a local tokenizer and token-ID prompts so lm-eval's continuation
boundary stays aligned with the endpoint's echoed logprobs. The runner allows custom
tokenizer code on that path (`trust_remote_code=True`); set
`extra_model_args.trust_remote_code: false` to disable it. Use
`extra_model_args.tokenizer_backend: huggingface` to load a local tokenizer for chat.

For local chat and completions endpoints, omit `temperature` and `seed` from
`gen_kwargs` to use the server's generation defaults. Evalchemy leaves those fields
out of requests even when a benchmark or lm-eval supplies its own defaults. Values
passed through `--gen_kwargs` take precedence for both custom and lm-eval tasks;
for example, `temperature=0.85,top_p=0.95,seed=27`. An explicit
`do_sample=false` requests greedy decoding. The evaluation `--seed` controls
local task randomness and does not set the endpoint's request seed.

## Telemetry

Telemetry is disabled unless a Finelog ingestion URL is provided:

```bash
uv run python -m eval.serve_eval.run --provider endpoint \
    --base-url http://localhost:8000/v1 \
    --telemetry-endpoint https://finelog.example/v1/telemetry \
    --root-run-uid qwen3-math \
    --execution-uid qwen3-math-attempt-1
```

`FINELOG_TELEMETRY_ENDPOINT`, `EVAL_ROOT_RUN_UID`, `EVAL_EXECUTION_UID`, and
`EVAL_SERVING_JOB_ID` are the environment equivalents. Keep `root_run_uid` stable for a
logical effort and use a new `execution_uid` for each invocation or retry; standalone
runs generate both independently. Optional `serving_job_id` joins Marin-owned
`service=vllm` records when the Iris job ID is already known.

The runner exports total evaluation `phase_duration_seconds`, terminal per-task and
total `work_completed` trials (`unit={item}`, `work_kind=trial`), and a run-level
`seconds_per_trial` only for a nonzero terminal total. Multiple tasks can share the child
process, so it does not claim per-task duration, per-request data, or live sample data.
Marin-serve teardown records the same phase metric with `phase=cleanup` and its outcome.

Marin must inject a reachable, authorized endpoint; Evalchemy does not discover Finelog
or credentials. Export and the two-second shutdown are best-effort and cannot change
evaluation output or exit status.

To gate a run's scores against a checked-in spec, hand the output dir to the gate:

```bash
uv run python -m eval.regression.validate check --results <output-dir> \
    --spec eval/regression/specs/qwen3-0.6b.json
```
