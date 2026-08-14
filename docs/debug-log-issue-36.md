# Debugging log for issue 36

Restore correct `lambada_openai` loglikelihood scoring through served OpenAI
completions endpoints.

## Initial status

`lambada_openai` scores collapsed when evaluated through a served vLLM OpenAI
completions endpoint, despite other tasks scoring normally. The runner supplied
`tokenized_requests=False` to every adapter.

## Hypothesis 1

With text requests, lm-eval calculates the context/continuation boundary with
its local Hugging Face tokenizer, then uses that count to slice the endpoint's
echoed logprob tokens. A server-side retokenization can move the boundary. This
is especially visible in LAMBADA, whose target begins with a leading space.

## Changes to make

Send token IDs for the `local-completions` adapter so the served endpoint
receives the exact sequence whose context length lm-eval calculated. Keep chat
requests as text because the chat-completions adapter accepts message objects.

## Results

The lm-eval 0.4.12 adapter sends token-ID prompts unchanged when
`tokenized_requests=True`, and its logprob parser slices the echoed sequence
using the same token count. The runner now selects that mode only for
`local-completions`. `tests/e2e/test_serve_eval.py` (29 passed) and
`tests/test_completion_responses.py` (39 passed) passed with the required
temporary `zstandard` test dependency.

## Future work

- [ ] Verify against a full served LAMBADA run in the next endpoint regression
  gate.
