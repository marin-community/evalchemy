# Olympiad Final-Answer Tasks

This suite provides four offline, deterministic task names:

- `RIMON`: 335 RIMO-N problems, graded by exact extracted-answer string match.
- `OlymMATHEasy`: 100 English OlymMATH-EASY problems.
- `OlymMATHHard`: 100 English OlymMATH-HARD problems.
- `AMOBenchParser`: the 39 number, set, and variable AMO-Bench problems that use
  the source's programmatic grader. The 11 description problems are intentionally
  excluded because they require an LLM judge.

OlymMATH reproduces the released Math-Verify scorer, including its normalized
string fallback after a verifier exception. Each sample records the scoring method
so fallback-accepted answers can be audited; the source fallback can accept substring
matches and should not be interpreted as a stricter equivalence check. AMO-Bench uses its source prompt and
ports the published parser logic, including the variable checks for question IDs 5
and 37. Its model-controlled SymPy solves run behind a terminating process timeout.

The base Evalchemy install includes the deterministic math grading dependencies.
To install these tasks explicitly through the per-benchmark extras contract, run:

```bash
uv sync --extra amobenchparser --extra olymmatheasy --extra olymmathhard --extra rimon
```

Every data directory contains a source manifest with a pinned revision and hash.
The AMO-Bench normalized subset can be reproduced with `prepare_data.py` from the
pinned source parquet. RIMO-P, OlymMATH Chinese/Lean splits, full AMO-Bench,
IMOAnswerBench, CHAMP, and EEFSUVA are outside this deterministic release.
