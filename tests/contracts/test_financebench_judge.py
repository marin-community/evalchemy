"""FinanceBench source and answer-equivalence judge parity."""

import json

from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark, PROMPT
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import evaluate_source


class FixedEndpoint:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return self.response


def test_financebench_preserves_document_prompt_and_equivalence_judge_plan(tmp_path):
    row = {"id": "filing-7", "question": "What was revenue?", "answer": "$10", "evidence_text": "Revenue was $10."}
    data_file = tmp_path / "financebench.jsonl"
    data_file.write_text(json.dumps(row) + "\n")
    source = FinanceBenchBenchmark(data_file=str(data_file))
    candidate = FixedEndpoint("$10")
    judge = FixedEndpoint("correct")

    metrics, samples = evaluate_source(
        source,
        SourceOptions("FinanceBench", SourceRef(str(data_file))),
        candidate,
        judges={"FinanceBench": judge},
    )

    assert metrics["accuracy"] == 1.0
    assert samples[0]["judge_label"] == "correct"
    assert candidate.requests[0].prompt == [{"role": "user", "content": PROMPT.format(**row)}]
    assert 'Reference answer(s):\n["$10"]' in judge.requests[0].prompt
