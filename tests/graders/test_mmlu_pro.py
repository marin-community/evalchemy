"""MMLU-Pro grading and sample records for reasoning model responses."""

from types import SimpleNamespace

from eval.chat_benchmarks.MMLUPro import eval_instruct as mmlu_pro
from eval.completion_response import CompletionContentPolicy, CompletionResponse, CompletionText
from eval.sample_logging import canonicalize_samples


def test_mmlu_pro_scores_final_answer_and_records_full_response(monkeypatch):
    examples = [
        {"question": "Choose B", "options": ["A", "B", "C"], "answer": "B", "category": "math", "cot_content": ""},
        {"question": "Choose C", "options": ["A", "B", "C"], "answer": "C", "category": "math", "cot_content": ""},
    ]
    monkeypatch.setattr(mmlu_pro, "load_dataset", lambda _name: {"test": examples, "validation": examples})
    benchmark = mmlu_pro.MMLUProBenchmark(num_fewshot=0)
    response = CompletionResponse(
        content="The answer is b.",
        reasoning_content="At first I thought the answer is A.",
        finish_reason="stop",
        usage=None,
        provider_metadata={},
        raw_choice={},
    )
    output = CompletionText(
        response.normalized_content(CompletionContentPolicy.COMBINE),
        response,
        CompletionContentPolicy.COMBINE,
    )
    plain_output = "<|start_think|>The answer is A.<|end_think|>The answer is C."
    model = SimpleNamespace(
        world_size=1,
        rank=0,
        apply_chat_template=lambda messages: messages[0]["content"],
        generate_until=lambda _requests: [output, plain_output],
    )

    generated = benchmark.generate_responses(model)
    scored = benchmark.evaluate_responses(generated)
    records = canonicalize_samples("MMLUPro", benchmark.to_samples(generated, scored))

    assert scored["accuracy_avg"] == 1.0
    assert [record["resps"] for record in records] == [[[output]], [[plain_output]]]
    assert [record["filtered_resps"] for record in records] == [["B"], ["C"]]
    assert records[0]["completion_responses"][0][0]["content"] == "The answer is b."
