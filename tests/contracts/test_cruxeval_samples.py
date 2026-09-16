import json

from eval.chat_benchmarks.CruxEval.eval_instruct import CruxEvalBenchmark
from eval.sample_logging import canonicalize_samples


class _CandidateModel:
    rank = 0
    world_size = 1

    def __init__(self):
        self.generated_ids = []

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, instances):
        self.generated_ids.extend(instance.idx for instance in instances)
        return ["[ANSWER] candidate [/ANSWER]" for _ in instances]


def test_cruxeval_sample_cap_emits_one_record_per_direction(tmp_path):
    rows = [
        {"id": f"sample_{index}", "code": "def f(x): return x + 1", "input": str(index), "output": str(index + 1)}
        for index in range(3)
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cruxeval.jsonl").write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    benchmark = CruxEvalBenchmark(data_dir=str(data_dir))
    benchmark.set_evaluation_limits(limit=1)
    model = _CandidateModel()

    generated = benchmark.generate_responses(model)
    scored = benchmark.evaluate_responses(generated)
    samples = benchmark.to_samples(generated, scored)
    canonical = canonicalize_samples("CruxEval", samples, benchmark.sample_manifest)

    assert model.generated_ids == ["sample_0", "sample_0"]
    assert [sample["sample_namespace"] for sample in canonical] == ["input", "output"]
    assert [sample["target"] for sample in canonical] == ["0", "1"]
    assert all(sample["doc_hash"] and sample["prompt_hash"] and sample["target_hash"] for sample in canonical)
