# Debugging log for short-answer-extraction

Restore short-answer scoring for Natural Questions and TriviaQA responses that
follow common answer forms without using the exact prompt marker.

## Initial status

`extract_marked_short_answer` only accepts a line beginning with `Answer:` or
`A:`. Correct responses such as `The answer is (Leeds).` and a final bare
answer after reasoning therefore become `[invalid]` before exact-match scoring.

## Hypothesis 1

The line-marker-only policy, not answer normalization or generation stops,
causes the rejected correct responses.

## Changes to make

Replace the scalar marker extractor with one reusable short-answer extraction
policy that returns both an answer and its format. Keep the strict contract
result as a separate filter, and use the flexible result for the task's main
exact-match score.

## Results

Before the change, common explicit and bare answer forms reproduce as
`[invalid]`.

The shared extractor now classifies contract, explicit, bare, and invalid
responses. Focused task-pipeline regression coverage passes for both NQ-Open
and TriviaQA, including nested answer markers and a bare final line after
reasoning.

The zstandard-enabled repository suite passes: 342 passed, 6 skipped.

## Future work

- [x] Verify structured extraction and both task filter pipelines with focused
  regression tests.
