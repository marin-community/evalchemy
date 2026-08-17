# Debugging log for issue #77 nightly on 2026-08-17

Restore the scheduled serve-and-eval lane after the TPU placement configuration
changed ahead of its Marin dependency.

## Initial status

Run 32007832029 failed before submitting an Iris job. Evalchemy passed
`v6e-4,v5litepod-4,v5p-8,v4-8` as one `marin-serve --tpu` value, and
`marin-core==0.2.84.dev32001502776` rejected it as an unknown TPU type.

## Hypothesis 1

Evalchemy PR #87 enabled compatible TPU alternatives before its dependency,
Marin PR #8345, was merged and published. The installed Marin CLI therefore
still accepts only one TPU slice type.

## Changes to make

Add a regression test that loads the shipped runner config and starts a fake
external `marin-serve` process implementing the released single-TPU interface.

## Results

The regression test failed against main: the fake released CLI exited with code
64 before serving because the shipped config contained a comma-separated value.

## Hypothesis 2

Using `v6e-4`, the first compatible topology selected by PR #87, preserves the
capacity workaround's primary hardware choice while satisfying the released
single-TPU CLI contract. Leaving the region unset still lets Iris place that
topology in any configured region.

## Changes to make

Set the runner default to `v6e-4`, document `--tpu` as singular, and align the
shipped config and public examples. Keep the workflow's blank scheduled input so
it inherits the runner default.

## Results

The regression passed with `v6e-4`. The provider reached the fake endpoint and
cleaned up the fake Iris job through the same subprocess and PTY boundaries used
by the nightly. The full hardware-free E2E harness passed: 54 tests in 55.35
seconds.

The failed run's cleanup step reported no matching Iris job. The CLI rejected
the TPU argument before submission, so no manual cleanup is required.

## Future work

- [ ] Restore compatible TPU alternatives after Marin PR #8345 is merged and the
  nightly's `marin-core` version floor includes the change.
