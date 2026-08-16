# Debugging log for humaneval-stop-cap

Restore HumanEval generation against OpenAI-compatible `/v1/completions`
servers and preserve its intended stop behavior.

## Initial status

HumanEval supplies six code-boundary stops plus twelve generic chat turn stops.
vLLM's OpenAI completions endpoint rejects the resulting eighteen-item `stop`
list because its API contract allows at most four strings.

## Hypothesis 1

HumanEval needs no more than four request stops: the scorer already truncates
the two omitted code boundaries and all generic chat turn boundaries after
generation.

## Changes to make

Make the HumanEval request stop list contain four code-boundary stops, and add
regression coverage for the OpenAI cap and post-generation truncation.

## Results

The request now sends four code-boundary stops, satisfying the OpenAI
completions limit. The scorer retains the full stop list and removes omitted
boundaries after generation. Focused regression coverage passes.

The zstandard-enabled repository suite passes: 343 passed, 6 skipped.

## Future work

- [x] Verify the request contains no more than four stops.
- [ ] Verify the reporter treats infrastructure-error markers as unscored.
