"""Sampling overrides observed by a local model through the shared runtime."""

import asyncio

import pytest

from eval.contracts.eval_spec import SourceRef
from eval.execution.backend import LMBackend
from eval.sources.common import SourceOptions, text_specs


@pytest.mark.parametrize(
    ("overrides", "expected_temperature"),
    [
        ({"temperature": 0.85, "do_sample": False}, 0.0),
        ({"temperature": 0.85, "do_sample": True}, 0.85),
        ({"temperature": 0.85}, 0.85),
        ({"temperature": 0.0}, 0.0),
    ],
)
def test_generation_overrides_respect_explicit_sampling_at_model_boundary(overrides, expected_temperature):
    class Model:
        world_size = 1

        def __init__(self):
            self.generation = []

        def apply_chat_template(self, messages):
            return messages[0]["content"]

        def generate_until(self, requests):
            self.generation.extend(request.args[1] for request in requests)
            return ["answer"] * len(requests)

    [spec] = text_specs(
        [{"id": "question"}], SourceOptions("sampling", SourceRef("fixture"), generation=overrides),
        prompt=lambda row: row["id"], target=lambda _row: "answer", grader="exact_match",
        generation={"do_sample": False, "temperature": 0.7},
    )
    model = Model()

    async def run():
        async with LMBackend(model) as backend:
            await backend.execute(spec.requests[0])

    asyncio.run(run())

    assert model.generation == [{"temperature": expected_temperature}]
