# Debugging log for GSM8K flexible extraction

Extract GSM8K final answers without treating LaTeX display delimiters as numbers.

## Initial status

GSM8K's flexible regex accepts `$` and `$$` without requiring a digit. Its
final-match selection therefore chooses a closing display delimiter after a
boxed answer.

## Hypothesis 1

A task-specific lm-eval filter can prioritize boxed and explicit final-answer
syntax, then use a digit-required numeric fallback. The normal lm-eval sample
record will retain the raw response in `resps` and the selected value in
`filtered_resps`.

## Changes to make

Register a GSM8K filter, select it from the task configuration before
`take_first`, and cover the configured pipeline with boxed, currency, and
delimiter-only responses.

## Results

Before the change, the real regex-plus-`take_first` pipeline returned `$$` for
`$$\\n\\boxed{18}\\n$$`.

The configured replacement pipeline selects `18` and retains the raw response
alongside the selected filtered value. It also keeps currency and signed or
comma-separated boxed numbers while rejecting delimiter-only input.
