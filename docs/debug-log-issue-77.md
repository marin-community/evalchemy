# Debugging log for issue #77

Preserve Iris diagnostics when the nightly's served endpoint does not become ready.

## Initial status

The August 14 nightly submitted its Iris job but timed out before the endpoint
registered. The workflow stopped the job before retaining its logs, so Marin #8271
cannot distinguish scheduling capacity from a v5e runtime failure.

## Hypothesis 1

The canonical Iris job ID is available to later workflow steps through `GITHUB_ENV`,
or can be recovered from `iris job list`, before the existing cleanup step runs.

## Changes to make

Collect `iris job logs` into the existing nightly artifact after a failed serving
step and before the cleanup step stops the job.

## Results

The workflow YAML parses, and the E2E harness passes with its required test
dependencies installed. The log command is best-effort so diagnostic collection
cannot replace the original failure. The next readiness timeout will validate the
artifact against a live Iris job.

The documented E2E CI dependency list currently omits `zstandard`, which its
telemetry test imports. Adding that test-only dependency makes the existing suite
pass; this issue's workflow change does not alter that CI command.

## Future work

- [ ] Use the retained logs to resolve Marin #8271 if the readiness timeout recurs.
