# Debugging log for task routing

Remove false task-not-found warnings for valid lm-evaluation-harness tasks while
preserving a pre-evaluation failure for genuinely unknown tasks.

## Initial status

`cli_evaluate` passed every selected task to the Evalchemy chat task manager
during annotator-model validation. Its capability query warned when a standard
lm-eval task was absent from the chat registry, although the lm-eval registry
had already resolved it.

## Hypothesis 1

Task routing is resolved separately by the annotator validation and evaluation
paths, allowing the chat manager to treat an lm-eval task as missing.

## Changes to make

Resolve selected tasks once after both registries are built, log the selected
registry, and use that route for annotator validation and evaluation dispatch.
Make the chat manager capability query return false for non-chat tasks without
logging a warning.

## Results

The initial focused-test command used a stale wheel in `.venv` and could not
import the new routing symbols. Reinstalling the local package with `uv sync
--reinstall-package evalchemy` exposed the current source. The focused routing
and sample-logging tests passed. The repository lint gate and full pytest suite
also passed.

## Future work

- [ ] None.
