# Opt-in Judge routes

FinanceBench, SimpleQA, SimpleQAMini, OlympiadBench and OlympiadBenchFull accept
`verifyit_enabled=True`. The disabled path retains the source graders and imports
no verifyit. Install the pinned `verifyit` extra and Judge's OpenAI dependency.

`verifyit_judge_policy="source_whole_label_nontext_empty_v1"` uses the source's
whole stripped, case-insensitive label contract. SimpleQA accepts exactly A/B/C;
equivalence accepts correct/incorrect/not_attempted. The explicitly selected
`legacy_lines_nontext_empty_v1` permits an explanatory prefix before a final label.
Both preserve the integration's nontext-to-empty candidate policy and zero credit
for empty candidates; the native path sends empty candidates to the provider.
Both use completion budgets 128 then 2048 on incomplete responses, no transport
retries, and at most 300 seconds per request. These transport choices differ from
the source retry schedule and are recorded with each prepared input.

`verifyit_timeout=300` bounds one complete scoring batch, including preparation,
Math, queued Judge work and incomplete-completion retries. The existing verifyit worker kills
its owned process group on termination. The deadline covers one task batch.

Olympiad's `source_first_box_dollar_alternatives_math_then_judge_v1` preparation
retains raw inputs, extracts the source first box and flattens dollar-group
reference alternatives. Math/BOXED and MAX compare every alternative; only
unresolved scored candidates reach a configured judge. Missing extraction scores
zero. Already extracted replay inputs use `extracted_answer`; normal responses
use `response`. Dataset populations remain source-owned.

Each example retains `verifyit_grade` or `verifyit_grades` with raw inputs,
effective policies and primitive verdicts. Accuracy uses primitive rewards;
source label fields and aggregate metric names remain available. Preparation,
provider and deadline failures abort the task. TaskOutcome has empty metrics and
preserves the finalized minimum verdict plus failure stage and category, keeping
invalid_task distinct from grader infrastructure. No failed component becomes a
successful zero-score task.

Install the client and its declared companion/core pins:

```sh
EVALCHEMY_REV="$(git rev-parse HEAD)"
uv venv .venv-judge --python 3.12
uv pip install --python .venv-judge/bin/python \
  "evalchemy[verifyit] @ git+https://github.com/marin-community/evalchemy@${EVALCHEMY_REV}"
```

To score saved SimpleQA responses through the public benchmark API, set
`JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY` for your OpenAI-compatible
provider, and save `{"examples": [{"question": "...", "answer": "...",
"model_output": "..."}]}` in `responses.json`. Then run:

```sh
.venv-judge/bin/python - <<'PY'
import json
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark

benchmark = SimpleQABenchmark(
    verifyit_enabled=True,
    verifyit_judge_policy="source_whole_label_nontext_empty_v1",
    verifyit_timeout=300,
)
with open("responses.json") as stream:
    responses = json.load(stream)
print(json.dumps(benchmark.evaluate_responses(responses)))
PY
```

The same three constructor options apply to FinanceBenchBenchmark,
SimpleQAMiniBenchmark, OlympiadBenchBenchmark and OlympiadBenchFullBenchmark in
their respective `eval.chat_benchmarks.<name>.eval_instruct` modules. Olympiad
saved responses use `problem`, `answer` and raw `model_output`; extracted-only
replays use `model_answer` and are recorded with their different input stage.
