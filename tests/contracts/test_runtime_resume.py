"""Durable replay of partial work through the shared item lifecycle."""

import asyncio
from dataclasses import replace

from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.finestore_output import FineStoreOutput, read_finestore_output
from eval.contracts.finestore_resume import FineStoreResumeManager
from eval.execution.engine import Evaluator
from eval.resume.fingerprint import RunFingerprint


def test_finestore_replays_partial_conversation_and_supersedes_scored_samples(tmp_path):
    root = str(tmp_path / "archive")
    spec = EvalSpec(
        task="conversation", source=SourceRef("fixture"), source_id="opaque/01", ordinal=0,
        requests=(
            ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "first"}]),
            ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "second"}], previous_turn=0),
        ),
        grader=GraderSpec("fixture"), doc={"question": "first"}, target="second answer",
    )
    tasks = ("conversation", "other")
    specs = [
        spec, replace(spec, source_id="opaque/02", ordinal=1, requests=spec.requests[:1]),
        replace(spec, task="other", source_id="opaque/03", requests=spec.requests[:1]),
    ]
    fingerprint = RunFingerprint(inputs={"fixture": "v1"})
    expected_result = {"results": {"conversation": {"accuracy": 0.5}, "other": {"accuracy": 0.0}}}

    class Backend:
        def __init__(self, interrupted=False):
            self.prompts = []
            self.interrupted = interrupted

        async def execute(self, request):
            self.prompts.append(request.prompt)
            if self.interrupted and len(self.prompts) == 2:
                raise ConnectionError("interrupted second turn")
            return "second answer" if len(request.prompt) == 3 else "first answer"

    async def grade(spec, outputs):
        return Grade({"accuracy": float(outputs[-1] == spec.target)})

    def run(backend):
        with FineStoreOutput(root, "run", tasks) as output:
            checkpoints = {task: FineStoreResumeManager(output.store, "run", task, fingerprint) for task in tasks}
            engine = Evaluator(backend, grade, output, workers=1, checkpoints=checkpoints)

            async def collect():
                return [result async for result in engine.run(specs)]

            completed = asyncio.run(collect())
            if all(item.error is None for item in completed):
                # Checkpoint finalization leaves the shared owner available for
                # publishing the final result after every source has finished.
                output.finish(expected_result)
            return completed

    first = run(Backend(interrupted=True))
    assert sum(result.error is not None for result in first) == 1
    assert ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 2
    assert read_finestore_output(root, "run") is None

    resumed = Backend()
    second = run(resumed)
    assert all(result.error is None for result in second)
    assert resumed.prompts == [[
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second"},
    ]]
    rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).to_pylist(maps_as_pydicts="strict")
    assert len(rows) == 3
    conversation = next(sample_from_archive_row(row) for row in rows if sample_from_archive_row(row).output == "second answer")
    assert [message.content for message in conversation.prompt_messages] == ["first", "first answer", "second"]
    with FineStoreOutput(root, "run", tasks) as output:
        assert len(list(FineStoreResumeManager(output.store, "run", "conversation", fingerprint).records())) == 3
        assert len(list(FineStoreResumeManager(output.store, "run", "other", fingerprint).records())) == 1
    assert read_finestore_output(root, "run") == expected_result
