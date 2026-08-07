# Debugging log for OlympiadBench aliases and uncertainty

Expose the legacy OlympiadBench subset separately from the full, pinned
text-only benchmark and report uncertainty appropriate to each evaluation.

## Initial status

`OlympiadBench` silently prefers a bundled 30-row JSONL. It produces one
accuracy, without standard error or reproducible source metadata. The bundled
file was added in evalchemy PR #24 as a hand-selected stratified sample from
`lmms-lab/olympiadbench[test_en]`; the PR did not record an immutable dataset
revision or a selection script.

## Hypothesis 1

Keeping `OlympiadBench` for the historical subset and registering a distinct
`OlympiadBenchFull` task lets existing comparisons remain interpretable while
new runs use the pinned full text-only source.

## Changes to make

- Keep the local 30-row subset behind `OlympiadBench` and document its source.
- Add `OlympiadBenchFull` backed by the immutable Hugging Face revision.
- Repeat the subset ten times and emit AIME24-style aggregate uncertainty.
- Record full-dataset provenance and single-run sample standard error.

## Results

The focused regression suite proves that both task aliases register, the
legacy subset repeats and reports the AIME24-style standard error, and the
full task requests the pinned source before excluding multimodal rows. Full
results carry the dataset identifier, revision, split, effective sample count,
and the single-run sample standard error.

`PYTHONPATH=. .venv/bin/python -m pytest tests/olympiadbench/test_dataset.py
tests/test_task_routing.py tests/test_evaluation_limits.py -q` passes 19 tests.
`scripts/ci/check_benchmark_extras.py OlympiadBenchFull` also passes in its
isolated, torch-free environment.

## Future work

- [ ] Rerun historical comparison sets with `OlympiadBenchFull`.
