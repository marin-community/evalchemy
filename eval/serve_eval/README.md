# Serve a model and run evalchemy against it

`eval.serve_eval.run` provisions a TPU via Marin's `marin-serve` (Iris → vLLM), or
attaches to an OpenAI-compatible server you already have, then runs `python -m
eval.eval` against the endpoint and prints each task's metrics. It is a thin
orchestrator over `eval.eval` — all of evalchemy's task/scoring/pass@k machinery runs
as-is.

The runner can also export bounded, best-effort lifecycle telemetry directly to
Finelog. Telemetry is inert unless a Finelog endpoint is configured, and it does not
add `marin-core` or local inference dependencies to the endpoint-only install.

This is the runner; it prints scores. Gating a run against a checked-in threshold spec
is the regression gate's job — see `eval/regression/`.

## Install

```bash
uv sync --no-dev --python 3.12 --extra serve-eval   # evalchemy + the runner (vLLM-free)

# marin-serve provider only -- an isolated tool, since marin-core can't co-resolve
# with evalchemy's deps:
uv tool install --prerelease allow "marin-core>=0.2.55.dev202607220801"
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
- `marin-serve` — provisions the TPU itself and returns a capability URL with a
  scoped token in the path, so no auth header or SSH tunnel is needed.

Chat models use `local-chat-completions` + a bare `--apply_chat_template` flag +
`tokenizer_backend=huggingface,tokenized_requests=False` (the served model tokenizes;
lm-eval keeps a HF tokenizer only for length bookkeeping).

## Telemetry

Pass the Finelog ingestion URL explicitly and give retries or externally orchestrated
invocations the same run identity:

```bash
uv run python -m eval.serve_eval.run --provider endpoint \
    --base-url http://localhost:8000/v1 \
    --telemetry-endpoint https://finelog.example/v1/telemetry \
    --run-id eval-2026-07-31-qwen3-math
```

`FINELOG_TELEMETRY_ENDPOINT` and `EVAL_RUN_ID` are the environment equivalents. If
`--run-id` is omitted, the runner creates a UUID for that invocation. The run ID, model,
provider, and task list are Finelog resource attributes; task and metric names remain
record attributes instead of being embedded in instrument names. Metric names therefore
do not repeat the `evalchemy` service name.

For a run outside the controller's private network, an Iris admin can mint a short-lived
capability for `/system/log-server` and pass the returned `url` with
`/v1/telemetry` appended. The capability URL is a credential: mask it in CI, do not put
it in logs, and keep its TTL to the run duration.

```bash
iris --cluster marin endpoints mint /system/log-server --ttl-hours 2
# returned url: https://iris.oa.dev/proxy/t/<token>/system.log-server/
capability_url="<url value printed by iris>"
export FINELOG_TELEMETRY_ENDPOINT="${capability_url%/}/v1/telemetry"
```

The parent runner exports provider and readiness timing, eval subprocess duration and
exit code, terminal run state, per-task terminal sample counts and numeric result
metrics, output/result persistence, cleanup outcomes, and structured failures with the
stage and exception type it observed. The runner gives queued records a two-second
total drain-and-shutdown budget. The `marin-rigging` telemetry queue, validation,
network, and shutdown paths are best-effort, so an invalid or unavailable endpoint
cannot change evaluation stdout or exit status.

The metric schema is intentionally small:

| Metric | Kind | Record attributes |
| --- | --- | --- |
| `provider_duration_seconds` | histogram | `provider`, `outcome` |
| `readiness_attempts` | counter | `provider`, `outcome` |
| `readiness_duration_seconds` | histogram | `provider`, `outcome` |
| `subprocess_duration_seconds` | histogram | `outcome` |
| `subprocess_exit_code` | gauge | snapshot attributes |
| `runs` | counter | `state` |
| `task_samples` | gauge | `task`, snapshot attributes |
| `task_metric` | gauge | `task`, `metric`, snapshot attributes |
| `results_persisted` | counter | none |
| `cleanup_duration_seconds` | histogram | `provider`, `outcome` |
| `cleanups` | counter | `provider`, `outcome` |

Snapshot gauges carry `source_kind=gauge` and
`source_temporality=current_snapshot`. Lifecycle details are structured events named
`run_started`, `provider_starting`, `provider_terminal`, `readiness_terminal`,
`subprocess_started`, `subprocess_terminal`, `output_ready`, `results_persisted`,
`cleanup_terminal`, `failure`, and `run_terminal`. Paths and error messages stay in
event bodies; process, task, metric, stage, and outcome dimensions are record
attributes.

The `eval.eval` child does not stream request or sample progress back to the parent.
Consequently this integration has no per-request latency/count records and no live
per-sample telemetry. It reports sample counts and metrics only after loading the
persisted `results_*.json`. Adding finer-grained telemetry requires an explicit child
process protocol or a separate Marin-owned instrumentation seam; this runner does not
infer those signals from subprocess output.

To gate a run's scores against a checked-in spec, hand the output dir to the gate:

```bash
uv run python -m eval.regression.validate check --results <output-dir> \
    --spec eval/regression/specs/qwen3-0.6b.json
```
