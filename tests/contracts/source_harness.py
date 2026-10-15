"""Exercise source items, registered graders, sample delivery, and reduction together."""

import asyncio

from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.execution.metrics import SourceMetrics
from eval.graders.specs import GRADERS


def evaluate_source(source, options, backend, *, judges=None, checkpoints=None, workers=2):
    sink = CollectSamples()
    grading = Graders(GRADERS)
    metrics = SourceMetrics(repeated=getattr(source, "n_repeat", 1) > 1, policy=source.REDUCER)
    engine = Evaluator(backend, grading, sink, workers=workers, judges=judges, checkpoints=checkpoints)

    async def run():
        async for completed in engine.run(source.iter_specs(options)):
            if completed.error is not None:
                raise completed.error
            metrics.add(completed.result)

    try:
        asyncio.run(run())
    finally:
        grading.close()
    return metrics.finish(options.task)["results"][options.task], sink.samples[options.task]
