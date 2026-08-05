END_OF_TURN_SEQUENCES = (
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


def truncate_at_stop(text: str, stops: tuple[str, ...] = END_OF_TURN_SEQUENCES) -> str:
    stop_indexes = [index for stop in stops if (index := text.find(stop)) >= 0]
    return text[: min(stop_indexes)] if stop_indexes else text
