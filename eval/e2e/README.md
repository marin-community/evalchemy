# Evalchemy end-to-end (e2e) test

A **one-shot** harness that exercises the whole evalchemy path against a *real*
served model:

1. **Serve** a small model (`Qwen/Qwen3-0.6B`) — via Marin's `marin-serve`
   (Iris controller → vLLM on a TPU slice), or any OpenAI-compatible endpoint.
2. **Evaluate** it with `eval.eval` over a small "sample query set."
3. **Gate** the results against a checked-in baseline and exit non-zero on
   regression.

```
serve (provider) ──► OpenAIEndpoint(/v1) ──► python -m eval.eval ──► results_*.json ──► gate vs baseline
```

## Install

```bash
uv sync --no-dev --python 3.11        # evalchemy base (vLLM-free), for the eval side

# For the marin-serve (provisioning) provider only -- installed as an ISOLATED
# tool because marin-core can't co-resolve with evalchemy's deps (see pyproject):
uv tool install --prerelease allow "marin-core>=0.2.0.dev0"
```

**marin-serve provider — workspace requirement.** `marin-serve` bundles its
current working directory as the Iris job workspace and the container `uv sync`s
it, so it must run from a **marin checkout** (a dir whose `pyproject.toml`
resolves `marin-core[tpu,vllm]`) — not the evalchemy checkout. Pass it with
`--marin-workspace <path>` (or `MARIN_WORKSPACE`). Keep that checkout free of
large untracked siblings (e.g. `.worktrees/`): the bundle is collected by
`rglob` minus `.gitignore`, so unrelated big directories can push it past the
25 MB limit. Both of these are marin-side sharp edges tracked upstream in
[marin-community/marin#7106](https://github.com/marin-community/marin/issues/7106);
until that lands, a marin checkout is required for the provisioning path. The
`endpoint` provider needs none of this.

## Quick start

The one-shot entrypoint is the Python module `eval.e2e.run_e2e` (`--help` for all
flags):

```bash
# A) Provision the accelerator AND run the eval (the default, one-shot path).
#    Brings up Qwen3-0.6B on a v5litepod-8 in europe-west4 via marin-serve mint
#    mode, evaluates, gates, and tears the Iris job down. Needs the marin-serve
#    tool + marin cluster creds + TPU quota + a marin checkout (see above).
uv run python -m eval.e2e.run_e2e --model Qwen/Qwen3-0.6B \
    --tpu v5litepod-8 --region europe-west4 --marin-workspace /path/to/marin

# B) Attach to a server you already have (no Marin dependency, runs anywhere):
E2E_BASE_URL=http://localhost:8000/v1 uv run python -m eval.e2e.run_e2e --provider endpoint

# Seed / refresh a baseline from a first real run (either provider):
uv run python -m eval.e2e.run_e2e --record-baseline
```

## Providers

| provider | how it serves | auth | needs | use |
|---|---|---|---|---|
| `marin-serve` *(the provisioning path)* | runs the real `marin-serve` CLI (Iris TPU job) as a background process in **`--access link` (mint) mode**, parses the printed capability `base_url`, and tears the Iris job down via `iris --cluster X job stop <id>` in a `finally` | **none** — the minted URL carries a scoped token in its path (`api_key` is any placeholder) | `marin-serve` on PATH (isolated `uv tool`), marin cluster creds, TPU quota, a marin checkout for `--marin-workspace` | provision-and-test, nightly / self-hosted CI |
| `endpoint` | attaches to an already-running `/v1` server | `--api-key` (optional, sent as `Authorization: Bearer`) | just a URL | unit tests, hosted-CI smoke, local dev, or a server someone already brought up |

The script **provisions the accelerator itself** with the `marin-serve` provider —
users do not have to stand up a server by hand (though they can, and point the
`endpoint` provider at it). The serving tool (`marin-core`/`marin-serve`) and the
eval stack (`evalchemy`) are **separate environments that only ever talk over the
HTTP endpoint** — `marin-serve` is never imported into evalchemy's interpreter,
so their (otherwise conflicting) dependency graphs never meet.

### Auth — mint mode solves it

`marin-serve --access link` mints a **public capability URL** with a scoped,
time-boxed token embedded in the path. Possession of the URL is the credential,
so a plain lm-eval OpenAI client reaches it off-cluster with **no IAP token, no
auth header, and no SSH tunnel** — `api_key` can be any non-empty placeholder.
The token authorizes only this endpoint and expires with the server
(`--timeout-hours`). This is the default (`provider.marin-serve.access: link`).

`--access private` restricts the proxy to cluster identity (IAP); use it only
when you additionally supply a valid bearer via `--api-key` / `E2E_API_KEY`.

## Configuration

`eval/e2e/config.yaml` holds the defaults (model, tasks, `limit`, decoding,
provider settings). Every field is overridable on the CLI. `gsm8k` is used
because it is **self-graded** (exact-match) — no judge model, no `OPENAI_API_KEY`
— so it is a clean generation smoke through the chat endpoint.

## The gate is a smoke check, not a regression baseline

A `--limit 20` run of a 0.6B model moves in coarse increments — gsm8k
`strict-match` alone swings ~3/20 run-to-run **even with greedy decoding** (vLLM/TPU
batching is not bit-exact). So by default the gate (`eval/e2e/compare.py`) asserts
only the two things that don't false-fail on that noise:

- **`expected_samples`** — the endpoint answered every query (the strongest
  connectivity signal);
- each metric **`>= min`** — a wide "the model isn't broken/empty" floor
  (`--record-baseline` sets `min = max(0.05, observed − 0.25)`).

A tight `reference ± tolerance` band would false-fail on the small-sample noise, so
`--record-baseline` no longer emits one; the observed values are kept under
`observed` for provenance. `compare.py` **still supports** an optional
`reference`/`tolerance` band if you want a tighter regression gate — add it by hand
and raise `--limit` so the metric is stable enough to justify it.

The baseline (`eval/e2e/baselines/qwen3-0.6b.json`) also carries a `provenance` block
(model/tokenizer revision, lm-eval version, decoding config, backend). **Pin
`model_revision` before committing floors** — the Hub tag is mutable. Regenerate
with `--record-baseline` after any intended change.

## Layout

| file | role |
|---|---|
| `run_e2e.py` | one-shot orchestrator (CLI) |
| `providers.py` | `endpoint` + `marin-serve` providers → `ServedModel(/v1, model, api_key)` |
| `eval_args.py` | build the `eval.eval` argv (mirrors Marin's `build_lm_eval_model_args`) |
| `compare.py` | gate `results_*.json` against a baseline |
| `config.yaml` | run defaults |
| `baselines/` | per-model baselines |
| `../../tests/e2e/` | unit tests (no model/Marin — hosted-CI safe) |
| `../../.github/workflows/e2e.yaml` | CI wiring |

## CI

`.github/workflows/e2e.yaml`:
- a **hosted** unit-test job (`tests/e2e/`) guards the harness on every push/PR;
- an optional **endpoint smoke** job runs when the repo variable `E2E_BASE_URL`
  is set;
- a **self-hosted** `marin-serve` job (nightly `schedule` + `workflow_dispatch`)
  provisions a TPU slice and runs the full cycle, with an unconditional cleanup
  step.
