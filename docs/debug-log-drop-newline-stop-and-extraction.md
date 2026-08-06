# Debugging log for DROP newline stopping and extraction

Prevent chat-completion formatting whitespace from ending DROP generations and
avoid replacing entity answers with incidental dates or numbers.

## Initial status

The DROP task config sends `"\\n"` as an OpenAI stop sequence. Chat models that
begin with formatting newlines produce empty visible completions. Separately,
the short-answer filter prefers dates and final numbers over unmarked entity
answers.

## Hypothesis 1

Use shared turn-boundary stops without a bare newline, prompt for an explicit
answer marker, and retain unmarked text unless a final entity span is reliably
identified. Extend completion quality validation to invalidate runs with mostly
empty normalized responses.

## Changes to make

Replace DROP's stop policy and prompt contract, revise the extraction fallback,
and add task-config, filter, and completion-quality regressions.

## Results

Before the change, the task config used a bare newline stop and the Chaz
Schilens completion extracted `20`. The new task configuration uses shared
turn-boundary stops and asks for `Answer: <short answer>`.

The filter selects marked numeric, date, entity, and multi-span answers. For an
unmarked verbose entity answer it selects the final name span, and it records
the extraction classification in sample artifacts. A successful run whose
normalized chat completions are all empty now sets the invalid-quality signal.
