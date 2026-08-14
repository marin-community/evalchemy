# Debugging log for issue #67

Expose portable evaluation-config validation before an evaluation starts.

## Initial status

`evalchemy` only accepts its legacy `--config` shape. It has no command that
loads the portable `evalchemy_config` schema and checks task names before model
construction.

## Hypothesis 1

The existing console entry point can dispatch a `validate-config` subcommand to
a dependency-light validator that loads the portable YAML and builds task-name
catalogs without constructing a model or downloading datasets.

## Changes to make

Add a validation CLI, route the existing entry point to it, and test valid YAML,
schema errors, and unknown task names through the public command boundary.

## Results

`evalchemy validate-config` now loads portable YAML, checks its task names against
the installed Evalchemy and lm-eval catalogs, and exits before model construction.
The console dispatcher stays outside `eval.eval`, so validation also works in the
dependency-light E2E harness. The public-command regression tests cover valid YAML,
an unknown task suggestion, and a forbidden schema field.

## Future work

- [ ] Keep the validator aligned with any future task-catalog changes.
