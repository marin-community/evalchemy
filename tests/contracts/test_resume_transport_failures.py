"""Resume coverage for typed endpoint responses."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from eval.completion_response import (  # noqa: E402
    CompletionContentPolicy,
    CompletionResponse,
    CompletionText,
    FailedGeneration,
)
from eval.resume import ManifestWriter, ResumeManager, RunFingerprint, read_manifest  # noqa: E402
from eval.sample_logging import canonicalize_samples  # noqa: E402
from eval.task import BaseBenchmark  # noqa: E402


class _Model:
    rank = 0
    world_size = 1

    def __init__(self, outputs):
        self.outputs = outputs

    def generate_until(self, instances):
        assert len(instances) == len(self.outputs)
        return self.outputs


class _Benchmark(BaseBenchmark):
    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError


def _fingerprint():
    return RunFingerprint.from_run_inputs(model_repo="model", task_name="task", task_data_digest="sha256:data")


def _instance():
    return Instance("generate_until", {"id": 0}, ("question", {"max_new_tokens": 8}), 0)


def test_resume_manifest_round_trips_typed_completion_metadata(tmp_path):
    response = CompletionResponse(
        content="final",
        reasoning_content="reasoning",
        finish_reason="stop",
        usage={"completion_tokens": 2},
        provider_metadata={"id": "response-1"},
        raw_choice={"index": 0},
    )
    output = CompletionText("reasoning\n\nfinal", response, CompletionContentPolicy.COMBINE)
    path = tmp_path / "manifest.jsonl"
    ManifestWriter(path).append(
        {"task": "task", "problem_idx": 0},
        {"outputs": [output, FailedGeneration("model_transport")]},
    )

    restored = read_manifest(path)[0].payload["outputs"]

    assert isinstance(restored[0], CompletionText)
    assert restored[0].artifact() == output.artifact()
    assert isinstance(restored[1], FailedGeneration)
    assert restored[1].failure_category == "model_transport"


def test_resumed_transport_failure_remains_classified_in_sample_artifact(tmp_path):
    first = _Benchmark()
    first.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="auto"))
    first.compute(_Model([FailedGeneration("model_transport")]), [_instance()])

    resumed = _Benchmark()
    resumed.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="auto"))
    restored = resumed.compute(_Model([]), [_instance()])[0]
    samples = resumed.to_samples(
        {"examples": [{"question": "question", "answer": "answer", "model_output": restored}]},
        {},
    )
    record = canonicalize_samples("FinanceBench", samples)[0]

    assert isinstance(restored, FailedGeneration)
    assert record["failure_category"] == "model_transport"
    assert record["completion_responses"][0][0]["failure_category"] == "model_transport"
