"""Streaming FineStore checkpoints preserve typed successful model responses."""

from eval.completion_response import CompletionContentPolicy, CompletionResponse, CompletionText, FailedGeneration
from eval.contracts.finestore_resume import FineStoreResumeManager
from eval.resume import RunFingerprint
from eval.runtime.checkpoints import CheckpointRow, ResumeStream, input_digest


def _fingerprint():
    return RunFingerprint(inputs={"task_name": "task", "model": "model"})


def test_finestore_stream_round_trips_typed_completion_metadata(tmp_path):
    response = CompletionResponse(
        content="final", reasoning_content="reasoning", finish_reason="stop", usage={"completion_tokens": 2},
        provider_metadata={}, raw_choice={},
    )
    output = CompletionText("reasoning\n\nfinal", response, CompletionContentPolicy.COMBINE)
    identity = {"source_id": "0"}
    first = FineStoreResumeManager(str(tmp_path), "run", "task", _fingerprint())
    first.record(CheckpointRow(0, input_digest(identity), {"phase": "generate"}, {"input_digest": "request", "output": output}))
    first.finalize()

    restored = ResumeStream(FineStoreResumeManager(str(tmp_path), "run", "task", _fingerprint())).admit(0, identity)
    value = restored.lookup({"phase": "generate"})["output"]

    assert isinstance(value, CompletionText)
    assert value.artifact() == output.artifact()


def test_failed_transport_response_is_not_checkpointed(tmp_path):
    manager = FineStoreResumeManager(str(tmp_path), "run", "task", _fingerprint())
    stream = ResumeStream(manager)
    journal = stream.admit(0, {"source_id": "0"})
    # Failed generations are intentionally retried on resume, never persisted as completed work.
    assert journal.lookup({"phase": "generate"}) is None
    assert isinstance(FailedGeneration("model_transport"), FailedGeneration)
