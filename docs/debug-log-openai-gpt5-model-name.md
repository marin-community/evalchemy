# Debugging log for OpenAI GPT-5 model-name matching

Keep configured stops and sampling controls for OpenAI-compatible provider aliases
that merely contain the character `5`.

## Initial status

lm-eval v0.4.12 treats every `OpenAIChatCompletion.model` containing `"5"`
as a GPT-5-family model. A served `qwen3.5-9b` alias therefore loses its stop
sequences and has temperature forced to 1.

## Hypothesis 1

An anchored matcher for the documented OpenAI `gpt-5`, `o1`, `o3`, and
`o4` model families preserves provider-specific requirements without mutating
unknown aliases.

## Changes to make

- Patch lm-eval's `OpenAIChatCompletion._create_payload` at Evalchemy startup.
- Cover GPT-5, Qwen3.5, and another non-OpenAI alias containing `5`.

## Results

The regression fails against lm-eval v0.4.12: `qwen3.5-9b` and
`provider-model-5` both lose their `stop` field. The current upstream main
revision also retains the same predicate, so there is no corrected revision to
pin.

Evalchemy now patches `OpenAIChatCompletion` with an anchored matcher for
`gpt-5`, `o1`, `o3`, and `o4` model families. GPT-5 still receives
OpenAI's fixed generation payload; aliases outside those families retain their
configured stops and temperature.

## Future work

- [ ] Advance the lm-eval pin when upstream publishes this correction.
