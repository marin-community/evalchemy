# Debugging log for humaneval-stop-cap

Restore HumanEval generation against OpenAI-compatible `/v1/completions`
servers and preserve its intended stop behavior.

## Initial status

HumanEval supplies six code-boundary stops plus twelve generic chat turn stops.
vLLM's OpenAI completions endpoint rejects the resulting eighteen-item `stop`
list because its API contract allows at most four strings.

## Hypothesis 1

HumanEval does not need generic chat turn stops at request time: its six
code-boundary stops are the upstream task contract, and the scorer already
truncates generic turn-boundary text after generation.

## Changes to make

Make the HumanEval request stop list contain only its code-boundary stops, and
add regression coverage for the OpenAI cap and post-generation turn truncation.

## Results

The request now sends four code-boundary stops, satisfying the OpenAI
completions limit. The scorer retains the full stop list and removes omitted
boundaries after generation. Focused regression coverage passes.

The zstandard-enabled repository suite passes: 343 passed, 6 skipped.

## Future work

- [x] Verify the request contains no more than four stops.
- [ ] Verify the reporter treats infrastructure-error markers as unscored.
