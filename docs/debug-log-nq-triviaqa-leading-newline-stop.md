# Debugging log for NQ-Open and TriviaQA leading-newline stops

Prevent NQ-Open and TriviaQA from sending bare newline stop sequences to
OpenAI-compatible chat endpoints.

## Initial status

The upstream task definitions pass `["\\n", ".", ","]` as server-side stops.
On the reported Qwen runs, formatting whitespace triggered the newline stop before
the endpoint returned reasoning or an answer. Evalchemy's run-level completion
quality guard correctly marks a majority of empty or reasoning-only-truncated chat
responses invalid, but cannot recover an answer that the server did not return.

## Hypothesis 1

Evalchemy's override include path can replace each task configuration while
preserving the upstream prompt, scoring, and whitespace filter, changing only the
unsafe stop list to the shared end-of-turn boundaries.

## Changes to make

- Add NQ-Open and TriviaQA override configurations under `eval/lm_eval_tasks/`.
- Exercise the real resolved configurations through the chat-completions adapter,
  including leading whitespace, punctuation, turn boundaries, and unusable
  completion classifications.

## Results

Before the overrides, the regression resolves NQ-Open and TriviaQA to lm-eval's
packaged configurations, whose generation stops are `["\n", ".", ","]`.
The test fails because neither task resolves to an Evalchemy override.

The overrides retain the upstream prompts, exact-match options, and
`remove_whitespace` filter, while replacing only the server-side stops with
`SHORT_ANSWER_STOP_SEQUENCES`, a mutable copy of the shared end-of-turn
boundaries required by lm-eval's task factory. The focused regression suite
passes 8 tests after the change.

lm-eval's default generation budget is 256 tokens. The 128-token budget in the
incident was supplied by the evaluation invocation, which overrides task
generation settings globally. There is no bounded endpoint available in this
workspace for a smoke run, so this change does not raise that caller-selected
budget. The existing completion-quality guard invalidates a run when at least
half its successful chat responses are empty or reasoning-only-truncated.

## Future work

- [ ] Run a bounded endpoint smoke before increasing a caller-supplied output budget.

## Hypothesis 2

After the safe stop fix, reasoning-capable chat models expose both their chain
of thought and their final content. The default `remove_whitespace` filter
passes that full transcript to exact-match scoring, so correct marked answers
cannot match the raw gold target.

## Changes to make

- Establish an `Answer: <short answer>` response contract in each task and its
  few-shot examples.
- Replace the whitespace-only filter with a shared flexible extractor that
  selects the last line-level answer marker and rejects unmarked transcripts.

## Results

The regression initially passes the combined reasoning transcript to exact-match
and fails on both NQ-Open and TriviaQA. The shared extractor now selects the
last line-level `Answer:`, `Final Answer:`, or legacy `A:` marker after
truncating a repeated turn boundary. Completions without a marker score as
`[invalid]`; the extractor does not search unmarked reasoning for a likely
answer.

Both tasks now prompt and few-shot with `Answer: <short answer>` while their
gold targets remain unchanged. The focused regression passes 16 tests,
including a reasoning-and-final chat response, punctuation, repeated turns,
and an unmarked transcript.
