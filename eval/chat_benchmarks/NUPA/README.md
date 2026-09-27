# NUPA

NUPA is the direct numeric question-answering benchmark from
[Number Cookbook: Number Understanding of Language Models and How to Improve It](https://arxiv.org/abs/2411.03766).
Evalchemy registers the full publication protocol as `NUPA` and a fixed
stratified convenience panel as `NUPA5K`.

## Dataset protocol

The benchmark downloads the canonical MIT-licensed
[`HaotongYang/NUPA_text`](https://huggingface.co/datasets/HaotongYang/NUPA_text)
test data at revision `01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7`.
The 688 MB source JSON contains 2,387,501 examples nested under 44 task keys and
2,391 task/digit groups.

Evalchemy reproduces the Number Cookbook text-model evaluation default: sample
100 examples from every task/digit group with Python's `random.Random(20222943)`.
The resulting benchmark contains 238,926 prompts. Two source groups contain
fewer than 100 examples; every other group contributes 100. The loader streams
one source task at a time, so it does not deserialize the complete source file
at once.

The source repository is GPL-3.0, so Evalchemy does not copy its implementation.
The dataset is separately distributed under the MIT license. The loader and
scorer here are clean-room implementations based on the paper's protocol and
observable metric definitions.

## Run the benchmark

Install the benchmark extra and evaluate an OpenAI-compatible endpoint:

```bash
uv sync --extra nupa

eval --model local-completions \
  --tasks NUPA \
  --limit 1000 \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

The full protocol contains 238,926 requests. Use `--limit` for development and
small comparisons; record the limit with any reported score. `--debug` uses four
checked-in records without downloading the source dataset.

For a cheaper comparison that retains coverage of every released task/digit
stratum, run `--tasks NUPA5K`. Its checked-in identity manifest selects 5,000
unique records by deterministic round-robin over the 2,391 strata, ordered by
task name and numeric digit length. Records within each stratum are ordered by
the SHA-256 digest of their exact source text. NUPA5K reports the ordinary
unweighted mean over the fixed panel; it is not an unbiased estimator of the
full publication protocol. Reported results must name `NUPA` or `NUPA5K` and
must not present the variants as interchangeable.

## Materialize the selected rows

The production loader reads the pinned nested source directly. The same
selection can be materialized as row-oriented JSONL for inspection:

```bash
uv run --extra nupa python -m eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset \
  --output /tmp/nupa_test.jsonl
```

Each row records its task, operation, answer representation, digit length,
S/M/L/XL length bucket, prompt, and target answer. `--num-each` and
`--random-seed` override the published protocol for experiments.

## Metrics

Numeric-component alignment follows the public NUPA text evaluation protocol.
Extraction also accepts final numbers in `\boxed{}`, inline math, or a trailing
equation after a thinking response. Numeric comparison remains representation-sensitive.
The benchmark reports:

- `exact_match`: representation-sensitive equality after format-specific extraction.
- `digit_match`: aligned digit accuracy between the extracted answer and target.
- `dlength`: absolute difference in total digit count; lower is better.
- `format_valid_rate`: fraction of responses accepted by the expected answer parser.
- `no_answer_rate`: fraction of responses without an extracted answer; lower is better.

Metrics are emitted overall, by task, by length bucket, and by task/bucket pair.
`exact_match` is the primary score.
