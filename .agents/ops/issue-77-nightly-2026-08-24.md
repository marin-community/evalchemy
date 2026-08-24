# Debugging log for issue 77 nightly failure on 2026-08-24

Diagnose Actions run 32704013188, restore the scheduled lane, and verify that
the failed run did not leave an Iris job consuming a TPU.

## Initial status

The `Serve, evaluate, and gate` step waited 1,800 seconds for
`/serve/evalchemy-e2e-qwen3-0-6b`, then exited before evalchemy sent any GSM8K
requests. The workflow log also reported that `iris job stop` was unavailable.

## Hypothesis 1

The serving task received a TPU but failed during model startup, so this is not
an evalchemy evaluation or regression-gate failure.

## Changes to make

Inspect the uploaded Iris task log and the exact TPU selected by Iris.

## Results

Confirmed. Iris selected a `v4-8` worker. Qwen3-0.6B loaded, then
`tpu-inference` failed during ragged-paged-attention warmup with
`NotImplementedError: Unsupported tpu_version=4`. The job became terminal after
4 minutes and 24 seconds, but `marin-serve` continued waiting for endpoint
registration until its 1,800-second client timeout.

## Hypothesis 2

The nightly configuration admits a TPU topology that its serving runtime cannot
use, and its cleanup command predates the Iris `job stop` to `job cancel` rename.

## Changes to make

Remove `v4-8` from the default compatible TPU alternatives and update both
provider and workflow cleanup to use `iris job cancel`. Add regression coverage
for the shipped topology set and the Iris cleanup boundary.

## Results

The three focused regression tests failed before the implementation and passed
after it. The complete hardware-free E2E suite and the changed-files lint pass
also pass. A read-only Iris query reports the runner job and its only task as
terminal `failed`, so the run holds no worker and needs no manual cancellation.

## Future work

- [x] Track `v4-8` serving support or early compatibility validation in
      [marin#7085](https://github.com/marin-community/marin/issues/7085).
