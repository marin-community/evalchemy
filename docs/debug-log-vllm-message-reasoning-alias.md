# Debugging log for vLLM message reasoning alias

Preserve vLLM Qwen chat-completion reasoning returned in `message.reasoning`.

## Initial status

The shared completion normalizer accepts `message.reasoning_content` and two
choice-level aliases, but not vLLM's `message.reasoning`. vLLM reasoning-only
responses are therefore classified as empty and scored as empty strings.

## Hypothesis 1

Resolving known aliases in priority order at the message boundary will give
lm-eval and Evalchemy-native benchmarks the same canonical reasoning text while
leaving the original provider response available in `raw_choice`.

## Changes to make

Add `message.reasoning` to the completion normalizer and cover every supported
reasoning layout, including the vLLM Qwen response shape.

## Results

Before the change, the `message.reasoning` cases failed in both task paths and
the Qwen-shaped response became an empty string. The existing aliases passed,
which isolates the regression to the missing message-level alias.

After adding the alias, all completion-response regressions pass. The raw
choice remains unchanged in the sample artifact while the canonical
`reasoning_content` field contains the vLLM value.
