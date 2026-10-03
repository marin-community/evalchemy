import pytest

from eval.chat_benchmarks.GraphWalks import eval_instruct as graphwalks
from eval.completion_response import (
    CompletionContentPolicy,
    CompletionText,
    FailedGeneration,
    completion_response_from_chat_choice,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, messages, *, enable_thinking=False, **kwargs):
        return {"input_ids": [0] * (len(messages[0]["content"].split()) + 2 + int(enable_thinking))}


class Model:
    rank = 0
    world_size = 1
    tokenizer = Tokenizer()
    _evalchemy_chat_template_kwargs = {"enable_thinking": True}

    def __init__(self, responses=None):
        self.responses = responses
        self.requests = []

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, instances):
        self.requests.extend(instances)
        return self.responses or ["Final Answer: [a, b]" for instance in instances]


def row(words=10, nodes=("a", "b"), kind="bfs"):
    prompt = "word " * words
    return {
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "answer_nodes": list(nodes),
        "problem_type": kind,
        "date_added": "2025-01-01",
    }


def benchmark(monkeypatch, rows, context=10000, cap=131072, limit=None):
    monkeypatch.setattr(graphwalks, "load_dataset", lambda *args, **kwargs: rows)
    result = graphwalks.GraphWalksBenchmark()
    result.set_evaluation_limits(max_length=context, max_tokens=cap, limit=limit)
    return result


@pytest.mark.parametrize(
    "response,nodes,f1,exact",
    [
        ("reasoning\nFinal Answer: [a, a, b]", ("a", "b", "c"), 0.8, 0.0),
        ("Final Answer: []", (), 1.0, 1.0),
        ("[a, b]", ("a", "b"), 0.0, 0.0),
        ("Final Answer: [a, b]\ntrailing prose", ("a", "b"), 0.0, 0.0),
    ],
)
def test_set_grading_preserves_last_line_contract(response, nodes, f1, exact):
    scores = graphwalks.grade_answer(response, nodes).scores
    assert scores["f1"] == f1
    assert scores["exact_match"] == exact


def test_generation_preserves_budgets_selection_and_opaque_source_identity(monkeypatch):
    rows = [row(2000), row(), row(nodes=tuple("node" for _ in range(100))), row()]
    task = benchmark(monkeypatch, rows, limit=2)
    model = Model()
    generated = task.generate_responses(model)
    scored = task.evaluate_responses(generated)
    samples = task.to_samples(generated, scored)
    assert [request.idx for request in model.requests] == [1, 2]
    assert [request.args[1]["max_new_tokens"] for request in model.requests] == [8200, 8396]
    assert [sample["doc_id"] for sample in samples] == [1, 2]
    assert samples[0]["arguments"][0][1]["max_new_tokens"] == 8200
    assert task.describe().n_benchmark == 4
    assert task.describe().n_attempted == 2
    assert generated["selection"]["skipped_context_by_type"] == {"bfs": 1}
    assert generated["selection"]["not_inspected_after_limit"] == 1
    assert scored["f1"] == 0.5
    assert samples[0]["f1"] == 1.0


def test_thinking_template_controls_context_boundary(monkeypatch):
    task = benchmark(monkeypatch, [row(10), row(9)], context=8276)
    generated = task.generate_responses(Model())
    assert [example["source_index"] for example in generated["examples"]] == [1]
    assert generated["selection"]["skipped_context_by_type"] == {"bfs": 1}


def test_output_cap_skips_without_shortening_gold_budget(monkeypatch):
    task = benchmark(monkeypatch, [row(nodes=tuple("node" for _ in range(100))), row(nodes=())], cap=8200)
    model = Model()
    generated = task.generate_responses(model)
    assert [request.idx for request in model.requests] == [1]
    assert generated["selection"]["skipped_output_cap_by_type"] == {"bfs": 1}
    assert model.requests[0].args[1]["max_new_tokens"] == 8198


def test_length_ended_parse_failure_scores_zero_and_retains_completion(monkeypatch):
    choice = {"message": {"content": "unfinished reasoning"}, "finish_reason": "length"}
    output = CompletionText(
        "unfinished reasoning", completion_response_from_chat_choice({}, choice), CompletionContentPolicy.COMBINE
    )
    task = benchmark(monkeypatch, [row()])
    generated = task.generate_responses(Model([output]))
    scored = task.evaluate_responses(generated)
    sample = task.to_samples(generated, scored)[0]
    assert scored["f1"] == 0.0
    assert scored["n_unanswered"] == 1
    assert sample["resps"][0][0].response.finish_reason == "length"
    assert sample["answer_extraction_errors"] == ["failed_to_parse"]


def test_transport_failures_fail_completion_gate(monkeypatch):
    task = benchmark(monkeypatch, [row(), row()])
    generated = task.generate_responses(Model(["Final Answer: [a, b]", FailedGeneration("transport")]))
    with pytest.raises(ValueError, match="scored 1/2"):
        task.evaluate_responses(generated)


def test_global_override_cap_preserves_selected_budget_in_request_and_artifact(monkeypatch):
    task = benchmark(monkeypatch, [row()], cap=32768)
    task.set_evaluation_generation_kwargs({"max_gen_toks": 16384, "temperature": 0.7})
    model = Model()
    generated = task.generate_responses(model)
    scored = task.evaluate_responses(generated)
    sample = task.to_samples(generated, scored)[0]
    assert model.requests[0].args[1]["max_new_tokens"] == 8200
    assert model.requests[0].args[1]["temperature"] == 0.7
    assert sample["arguments"][0][1] == {"max_new_tokens": 8200, "temperature": 0.7, "do_sample": True}
