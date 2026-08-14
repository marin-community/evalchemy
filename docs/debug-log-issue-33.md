# Debugging log for issue 33

Restore the CruxEval evaluator import in the installed Evalchemy package.

## Initial status

`import eval.chat_benchmarks.CruxEval.evaluation` fails with
`ModuleNotFoundError: No module named 'execution'`. The top-level benchmark
module already uses a relative import, but its sibling evaluator does not.

## Hypothesis 1

`evaluation.py` imports `execution.py` as a top-level module. That works only
when a loader happens to put the benchmark directory on `sys.path`; it fails for
the package import path users and integrations use.

## Changes to make

Import `check_correctness` from the sibling module relatively and cover the
installed package import.

## Results

The package-import regression test passes. The lean-install check also imports
CruxEval successfully without torch or vLLM.

## Future work

- [ ] None observed.
