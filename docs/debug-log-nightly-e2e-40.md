# Debugging log for nightly e2e issue 40

Diagnose GitHub Actions run 29908091066 and restore the scheduled serve-and-eval lane.

## Initial status

The July 22 scheduled run exited before producing results. The cleanup step found no
Iris job named `evalchemy-e2e-qwen3-0-6b`.

## Hypothesis 1

The evalchemy provider invokes an obsolete `marin-serve` command after
[Marin PR #7452](https://github.com/marin-community/marin/pull/7452) changed the
CLI from `marin-serve MODEL` to `marin-serve iris MODEL`.

## Changes to make

Add a boundary test that records the argv received by a fake `marin-serve` process.
The process will emit the normal ready output and point the provider at a local HTTP
stub, so the test exercises provider startup and readiness without an Iris cluster.

## Results

The boundary test received `Qwen/Qwen3-0.6B` as argv element zero, followed by
the removed `--access link` option. This reproduces the failed run against the new
grouped CLI.

## Hypothesis 2

Inserting the `iris` subcommand and removing `--access` will match the new CLI.
The provider can retain its `access` setting for choosing which emitted URL to use:
the current Iris command always prints both the cluster-only and capability URLs.

## Changes to make

Update `MarinServeProvider` to invoke `marin-serve iris MODEL`, remove the obsolete
option, and raise the documented and nightly Marin floor to the first version
observed with that contract.

## Results

The focused boundary test passed after the command update. The complete hardware-free
e2e suite passed with 33 tests. Running the pinned tool's help confirmed that
`marin-serve iris` accepts every option the provider supplies and has no `--access`
option.

## Future work

- [ ] Confirm the next scheduled or manually dispatched cluster run is green before
  closing issue #40.
