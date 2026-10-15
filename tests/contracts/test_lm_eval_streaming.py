"""Native filter variants reach FineStore before the aggregate commit."""

import asyncio
from functools import partial

from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.finestore_output import FineStoreOutput
from eval.execution.engine import Evaluator
from eval.execution.grading import Grader, Graders
from eval.sources.lm_eval import grade_lm_eval


class Filters:
    def apply(self, instances):
        for instance in instances:
            instance.filtered_resps.update(strict="wrong", flexible="4")


class Task:
    _filters = [Filters()]

    def process_results(self, doc, responses):
        return {"exact_match": float(responses[0] == "4")}


class Backend:
    async def execute(self, request):
        return "4"


def test_native_filter_variants_are_durable_before_result_commit(tmp_path):
    root = str(tmp_path / "archive")
    spec = EvalSpec("gsm8k", SourceRef("fixture"), "opaque", 0,
                    (ModelRequest(RequestKind.GENERATE, "2 + 2?"),),
                    GraderSpec("native", {"repeats": [1], "indices": [0]}),
                    doc={"question": "2 + 2?"}, target="4")
    grading = Graders({"native": Grader(partial(grade_lm_eval, Task()))})
    with FineStoreOutput(root, "run", ["gsm8k"]) as output:
        engine = Evaluator(Backend(), grading, output, workers=1)
        async def run():
            async for completed in engine.run([spec]):
                assert completed.error is None
        try:
            asyncio.run(run())
        finally:
            grading.close()
        staged = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        samples = {row["filter"]: sample_from_archive_row(row) for row in staged.to_pylist(maps_as_pydicts="strict")}
        assert {name: sample.correct for name, sample in samples.items()} == {"strict": False, "flexible": True}
