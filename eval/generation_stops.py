"""Shared stop policies for generation and scoring."""

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
# lm-eval imports these through its ``!function`` YAML tag and requires ``until``
# to be a list. Its task factory deep-copies the config before models may append EOS.
GSM8K_STOP_SEQUENCES: list[str] = [  # noqa: ml-module-globals
    "Question:" if stop == "\nQuestion:" else stop for stop in END_OF_TURN_SEQUENCES
]
DROP_STOP_SEQUENCES: list[str] = list(END_OF_TURN_SEQUENCES)  # noqa: ml-module-globals
SHORT_ANSWER_STOP_SEQUENCES: list[str] = list(END_OF_TURN_SEQUENCES)  # noqa: ml-module-globals
HUMANEVAL_STOP_SEQUENCES: list[str] = [  # noqa: ml-module-globals
    "\nclass",
    "\ndef",
    "\n#",
    "\nif",
    "\nprint",
    "\n```",
    *END_OF_TURN_SEQUENCES,
]
# OpenAI-compatible completions endpoints accept at most four request stops.
# The scorer still applies the complete set after generation, so omitting the
# remaining boundaries from the request cannot leak text into a prediction.
HUMANEVAL_REQUEST_STOP_SEQUENCES: list[str] = HUMANEVAL_STOP_SEQUENCES[:4]  # noqa: ml-module-globals


def truncate_at_stop(text: str, stops: Sequence[str] = END_OF_TURN_SEQUENCES) -> str:
    stop_indexes = [index for stop in stops if (index := text.find(stop)) >= 0]
    return text[: min(stop_indexes)] if stop_indexes else text
