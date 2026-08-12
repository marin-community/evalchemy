# NUPA

NUPA is the direct numeric question-answering benchmark from
["Number Cookbook: Number Understanding of Language Models and How to Improve It"](https://arxiv.org/abs/2411.03766).
Evalchemy registers it as one native task named `NUPA`.

The integration has two stages:

1. `data_prep/flatten_hf_dataset.py` converts the original nested dataset once
   and publishes row-oriented records to Hugging Face.
2. `NUPABenchmark` loads those records, requests model completions, scores each
   response, and aggregates the metrics.

## Dataset repositories

The conversion source is the MIT-licensed
[`HaotongYang/NUPA_text`](https://huggingface.co/datasets/HaotongYang/NUPA_text)
dataset. The original source has nested task and digit mappings, so it is not
loaded directly during evaluation.

The flattened dataset repository is currently `TODO_ORG/nupa-text-eval`.
This identifier is a placeholder shared by the conversion command and runtime
loader. Finalize the owning Hugging Face organization and repository name before
publishing the production conversion or merging the integration. Update
`PUBLISHED_DATASET_NAME` in `eval_instruct.py` when the repository is chosen.

The flattened schema is:

```json
{
  "id": "test:max_Float_Float_Float:3:000000",
  "task_name": "max_Float_Float_Float",
  "operation": "max",
  "answer_format": "Float",
  "digit": 3,
  "length_bucket": "S",
  "prompt": "Directly return ... Get the maximal number: 9.11 and 9.9 =",
  "answer": "9.9"
}
```

`answer_format` is one of `Integer`, `Float`, `Fraction`, or
`ScientificNotation`. `length_bucket` is one of `S`, `M`, `L`, or `XL`.

## Convert and publish

Install the benchmark dependency and authenticate the Hugging Face CLI before
publishing:

```bash
uv sync --extra nupa
hf auth login
```

Download the original `test.json`, stream-flatten it, and publish the result:

```bash
uv run python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \
  --dataset-name HaotongYang/NUPA_text \
  --split test \
  --output /tmp/nupa_test.jsonl \
  --repo-id TODO_ORG/nupa-text-eval
```

The converter records the source dataset revision in the published dataset card.
It reads one top-level task at a time and writes JSONL incrementally; it does not
hold the complete nested source or flattened result in memory.

For a publishing smoke test, retain one example from every task-and-digit group:

```bash
uv run python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \
  --dataset-name HaotongYang/NUPA_text \
  --split test \
  --limit-per-task-digit 1 \
  --output /tmp/nupa_test_smoke.jsonl \
  --repo-id USER/nupa-text-eval-smoke
```

The smoke dataset checks conversion coverage and upload behavior. Do not report
benchmark performance from it.

## Run the benchmark

Evaluate the published dataset against an OpenAI-compatible endpoint:

```bash
eval --model local-completions \
  --tasks NUPA \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

Use `--debug` to load the four checked-in smoke records instead of Hugging Face:

```bash
eval --model local-completions \
  --tasks NUPA \
  --debug \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

## Scoring and metrics

Response extraction and normalization follow the observable behavior of the
official NUPA text evaluator. Evalchemy's scorer is a clean-room implementation;
the Number Cookbook code repository is GPL-3.0 and its code is not copied here.

The benchmark reports:

- `exact_match`: representation-sensitive equality after format-specific
  extraction and normalization.
- `digit_match`: aligned digit accuracy between the extracted answer and target.
- `dlength`: absolute difference in total digit count; lower is better.
- `format_valid_rate`: fraction of responses accepted by the expected answer
  format parser.
- `no_answer_rate`: fraction of responses from which no answer was extracted;
  lower is better.
- `dataset_num_samples`: number of evaluated rows.

Metrics are emitted overall and under these prefixes:

```text
task:<task_name>/
bucket:<length_bucket>/
task:<task_name>/bucket:<length_bucket>/
```

The task key is grouping metadata, not an Evalchemy task. One `NUPA` evaluation
runs dataset rows from multiple task-family and representation combinations.
