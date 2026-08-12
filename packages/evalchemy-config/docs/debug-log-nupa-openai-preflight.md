# Debugging log for NUPA OpenAI preflight

Allow native benchmarks to use the supported `openai-chat-completions` adapter
without requiring a tokenizer that the adapter intentionally does not provide.

## Initial status

The NUPA debug run fails before transport because endpoint preflight calls
`apply_chat_template` on a `None` tokenizer.

## Hypothesis 1

The OpenAI chat adapter defaults to no tokenizer, but bounded native generation
unconditionally tokenizes its payload during context preflight.

## Changes to make

Add a regression test requiring preflight to preserve generation arguments and
skip token counts when no tokenizer is configured.

## Results

Inspection confirmed `OpenAIChatCompletion(tokenizer_backend=None)` passes a
`None` tokenizer to `preflight_endpoint_generation`.

The regression test failed with the reported `NoneType.apply_chat_template`
error. Preflight now treats a missing tokenizer like a missing context limit:
it preserves the requested generation arguments and skips local token counting.

## End-to-end check

A 40-example NUPA run completed through `openai-chat-completions` after the
preflight change. It produced 40 model responses without infrastructure-error
markers. This run used a limited staging dataset and is evidence for integration
behavior, not benchmark performance.
