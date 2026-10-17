"""Replacing inference transport preserves committed request recovery."""

import argparse
import asyncio
from dataclasses import replace

import pytest
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.finestore_resume import ResumeRefused
from eval.execution.engine import Evaluator
from eval.execution.run_spec import RunSpec
from eval.resume.wiring import build_resume_wiring


class EndpointBackend:
    def __init__(self, template="template", fail_second=False):
        self.template = template
        self.fail_second = fail_second
        self.prompts = []

    def chat_template(self, _apply_chat_template):
        return self.template

    async def execute(self, request):
        self.prompts.append(request.prompt)
        if self.fail_second and len(self.prompts) == 2:
            raise ConnectionError("inference endpoint lost")
        return f"answer to {request.prompt}"


def wiring_args(endpoint, num_concurrent=16, **overrides):
    values = dict(
        model_args=f"model=Qwen/Qwen3-8B,revision=abc123,base_url={endpoint},num_concurrent={num_concurrent}",
        model="local-completions", include_path=None, resume_mode="auto",
    )
    return argparse.Namespace(**{**values, **overrides})


def test_replacement_endpoint_executes_only_unfinished_requests(tmp_path):
    root = str(tmp_path / "archive")
    specs = [EvalSpec(
        task="endpoint", source=SourceRef("fixture"), source_id=str(index), ordinal=index,
        requests=(ModelRequest(RequestKind.GENERATE, prompt),),
        grader=GraderSpec("fixture"), doc={"question": prompt}, target=f"answer to {prompt}",
    ) for index, prompt in enumerate(("first", "second"))]

    async def grade(spec, responses):
        return Grade({"accuracy": float(responses[0] == spec.target)})

    def run(endpoint, backend, num_concurrent):
        with FineStoreOutput(root, "run", ["endpoint"]) as output:
            factory = build_resume_wiring(
                wiring_args(endpoint, num_concurrent), backend, RunSpec(), store=output.store,
            )
            engine = Evaluator(backend, grade, output, workers=1, checkpoints={"endpoint": factory("endpoint")})

            async def collect():
                return [item async for item in engine.run(specs)]

            return asyncio.run(collect())

    interrupted = run("http://old-server/v1/completions", EndpointBackend(fail_second=True), 16)
    assert sum(item.error is not None for item in interrupted) == 1
    replacement = EndpointBackend()
    recovered = run("http://replacement-server/v1/completions", replacement, 32)
    assert replacement.prompts == ["second"]
    assert [item.result.grade.metrics for item in recovered] == [{"accuracy": 1.0}, {"accuracy": 1.0}]


@pytest.mark.parametrize("change", ["model", "revision", "adapter", "template", "generation"])
def test_replacement_endpoint_refuses_material_changes(tmp_path, change):
    with FineStoreOutput(str(tmp_path / "archive"), "run", ["endpoint"]) as output:
        original = build_resume_wiring(
            wiring_args("http://old-server/v1/completions"), EndpointBackend(), RunSpec(), store=output.store,
        )
        assert original("endpoint").decide() == "fresh"
        args = wiring_args("http://replacement-server/v1/completions")
        backend, run = EndpointBackend(), RunSpec()
        if change == "model":
            args.model_args = args.model_args.replace("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
        elif change == "revision":
            args.model_args = args.model_args.replace("abc123", "def456")
        elif change == "adapter":
            args.model = "local-chat-completions"
        elif change == "template":
            backend.template = "different template"
        else:
            run = replace(run, generation={"temperature": 0.5})
        replacement = build_resume_wiring(args, backend, run, store=output.store)
        with pytest.raises(ResumeRefused):
            replacement("endpoint").decide()
