from eval.generation_stops import END_OF_TURN_SEQUENCES, truncate_at_stop

HUMANEVAL_STOP_SEQUENCES = (
    "\nclass",
    "\ndef",
    "\n#",
    "\nif",
    "\nprint",
) + END_OF_TURN_SEQUENCES


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
):
    from lm_eval.tasks.humaneval.utils import pass_at_k as upstream_pass_at_k

    return upstream_pass_at_k(references, predictions, k)


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [
            doc["prompt"] + truncate_at_stop(response, HUMANEVAL_STOP_SEQUENCES)
            for response in responses
        ]
        for responses, doc in zip(resps, docs)
    ]
