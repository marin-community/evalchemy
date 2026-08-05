from eval.generation_stops import HUMANEVAL_STOP_SEQUENCES, truncate_at_stop


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
):
    # lm_eval's HumanEval module executes a code-eval probe at import time.
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
