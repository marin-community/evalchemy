# Regression gate for evalchemy runs

`eval.regression.validate` gates an eval run's scores against a checked-in spec, or
records a new one. It reads a run's `results_*.json` (produced by any evalchemy run —
typically the `eval/serve_eval` runner) and a spec file.

```bash
# Gate a run against the spec (exit 1 on regression):
uv run python -m eval.regression.validate check  --results <run-dir> --spec specs/qwen3-0.6b.json

# Seed a new spec from a real run:
uv run python -m eval.regression.validate record --results <run-dir> --spec specs/qwen3-0.6b.json
```

`--results` accepts a `results_*.json` file or a run dir to search. `--spec` defaults
to `specs/qwen3-0.6b.json`.

## The gate

The gate is a connectivity + coarse-quality smoke check. gsm8k `strict-match` swings a
few samples run-to-run at `--limit 20` even greedy (vLLM/TPU batching varies), so
`validate.py` checks two things that tolerate that noise:

1. the endpoint answered `expected_samples` queries (a low count means the endpoint
   dropped requests), and
2. each gated metric clears a floor — `record` sets `min = max(0.05, observed −
   margin)` (margin defaults to 0.25), a wide "model isn't broken/empty" floor.

`MetricThreshold` also accepts an optional two-sided `reference ± tolerance` band for a
tighter gate at higher `--limit`; use it only where run-to-run variance is small, since
a two-sided band fails on score *improvements* too.

Provenance (model, tokenizer, adapter, limit, seed, lm-eval version) is recorded for
the human reading a spec; the gate does not currently enforce comparability. Pin
`provenance.model_revision` before committing floors — the Hub tag is mutable.
