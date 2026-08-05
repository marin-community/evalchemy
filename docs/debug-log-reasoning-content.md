# Debugging log for reasoning content

Preserve reasoning-only OpenAI-compatible chat completions so lm-eval does not
score them as empty answers.

## Initial status

`LocalChatCompletion.parse_generations` reads only `message.content`. When it
is null, lm-eval replaces the result with its empty none-answer placeholder,
even when `message.reasoning_content` is non-empty.

## Hypothesis 1

Normalizing an OpenAI chat choice at the shared lm-eval adapter boundary can
preserve reasoning for both lm-eval tasks and Evalchemy-native benchmarks that
call the same model adapter.

## Changes to make

Add a typed response normalizer and install it on the local chat-completions
adapter. Make combined reasoning and final content the default, with an
explicit final-content-only policy.

## Results

The pre-fix regression returned `None` for non-empty `reasoning_content` when
the final content was null.

The shared normalizer retains content, reasoning content, finish reason, usage,
provider metadata, and the raw choice. It propagates those fields to
`--log_samples` artifacts, classifies reasoning-only length responses separately,
and emits a run-level invalid-quality signal when they make up at least half of
the run's observed chat completions.

The focused completion and generation-stop tests passed. The full local pytest
suite and `infra/pre-commit.py --all-files --fix` also passed.
