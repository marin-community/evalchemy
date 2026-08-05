"""Stop policies shared by Python benchmarks and declarative lm-eval tasks.

Despite its name, lm-eval's ``!function`` YAML tag imports any module attribute,
so task configs can reference these lists directly.
"""

from collections.abc import Sequence

END_OF_TURN_SEQUENCES: tuple[str, ...] = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)
GSM8K_STOP_SEQUENCES: list[str] = [
    "Question:" if stop == "\nQuestion:" else stop
    for stop in END_OF_TURN_SEQUENCES
]
HUMANEVAL_STOP_SEQUENCES: list[str] = [
    "\nclass",
    "\ndef",
    "\n#",
    "\nif",
    "\nprint",
    "\n```",
    *END_OF_TURN_SEQUENCES,
]


def truncate_at_stop(text: str, stops: Sequence[str] = END_OF_TURN_SEQUENCES) -> str:
    stop_indexes = [index for stop in stops if (index := text.find(stop)) >= 0]
    return text[: min(stop_indexes)] if stop_indexes else text
