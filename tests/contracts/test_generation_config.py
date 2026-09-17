"""Generation settings observed at an OpenAI-compatible endpoint."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from lm_eval.tasks import TaskManager

from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.contracts.sample_manifest import SampleManifest
from eval.resume.lm_eval_native import resume_simple_evaluate
from eval.robust_api import configure_generation_overrides


class _Endpoint(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(json.loads(body))
        completion = {"index": 0, "text": "\\boxed{42}", "message": {"content": "\\boxed{42}"}}
        response = json.dumps({"choices": [completion]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


@pytest.fixture
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Endpoint)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _model(endpoint, model_class):
    path = "chat/completions" if model_class is LocalChatCompletion else "completions"
    return model_class(
        model="served",
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1/{path}",
        tokenizer_backend=None,
        tokenized_requests=False,
    )


@pytest.mark.parametrize("model_class", [LocalChatCompletion, LocalCompletionsAPI])
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, {}),
        ({"do_sample": False}, {"temperature": 0}),
        ({"top_p": 0.91}, {"top_p": 0.91}),
        ({"temperature": 0.85, "seed": 27, "top_p": 0.91}, {"temperature": 0.85, "seed": 27, "top_p": 0.91}),
    ],
)
def test_native_generation_respects_caller_overrides_at_endpoint(endpoint, model_class, overrides, expected):
    model = _model(endpoint, model_class)
    prompt = [{"role": "user", "content": "Question"}] if model_class is LocalChatCompletion else "Question"
    request = Instance(
        "generate_until",
        {"question": "Question"},
        (prompt, {"temperature": 0.0, "top_p": 0.4, "do_sample": False}),
        0,
    )
    request.task_name = "example"
    request.doc_id = 0
    request.repeats = 1

    def evaluate_one_request(**kwargs):
        return kwargs["model"].generate_until([request])

    assert resume_simple_evaluate(
        evaluate_one_request,
        model=model,
        tasks=["example"],
        gen_kwargs=overrides,
        sample_manifest=SampleManifest("example"),
    ) == ["\\boxed{42}"]
    payload = endpoint.requests[0]
    for key in ("temperature", "seed"):
        if key in expected:
            assert payload[key] == expected[key]
        else:
            assert key not in payload
    if "top_p" in expected:
        assert payload["top_p"] == expected["top_p"]


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0.85, "seed": 27, "top_p": 0.91}])
def test_math500_generation_respects_caller_overrides_at_endpoint(endpoint, tmp_path, overrides):
    data_file = tmp_path / "math.jsonl"
    data_file.write_text(json.dumps({"problem": "What is 6 times 7?", "answer": "42"}) + "\n")
    benchmark = MATH500Benchmark(data_file=str(data_file))
    benchmark.set_evaluation_generation_kwargs(overrides)
    model = _model(endpoint, LocalChatCompletion)
    configure_generation_overrides(model, overrides)

    result = benchmark.generate_responses(model)

    assert result["examples"][0]["model_output"] == "\\boxed{42}"
    payload = endpoint.requests[0]
    for key in ("temperature", "seed"):
        if key in overrides:
            assert payload[key] == overrides[key]
        else:
            assert key not in payload
    if "top_p" in overrides:
        assert payload["top_p"] == overrides["top_p"]
    else:
        assert "top_p" not in payload


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0.85, "seed": 27, "top_p": 0.91}])
def test_lm_eval_task_generation_respects_caller_overrides_at_endpoint(endpoint, tmp_path, overrides):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({"question": "What is 6 times 7?", "answer": "\\boxed{42}"}) + "\n")
    task = {
        "task": "generation_config_probe",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(dataset)}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{question}}",
        "doc_to_target": "{{answer}}",
        "metric_list": [{"metric": "bypass"}],
        "generation_kwargs": {"temperature": 0.0, "top_p": 0.4, "do_sample": False},
    }
    (tmp_path / "generation_config_probe.yaml").write_text(yaml.safe_dump(task))

    results = resume_simple_evaluate(
        evaluator.simple_evaluate,
        model=_model(endpoint, LocalChatCompletion),
        tasks=["generation_config_probe"],
        task_manager=TaskManager(include_path=str(tmp_path)),
        sample_manifest=SampleManifest("generation_config_probe"),
        gen_kwargs=overrides,
        apply_chat_template=True,
        limit=1,
    )

    assert "generation_config_probe" in results["results"]
    payload = endpoint.requests[0]
    for key in ("temperature", "seed"):
        if key in overrides:
            assert payload[key] == overrides[key]
        else:
            assert key not in payload
    assert payload["top_p"] == overrides.get("top_p", 0.4)


def test_cli_generation_config_wins_over_math500_defaults(endpoint, tmp_path):
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.eval",
            "--model",
            "local-chat-completions",
            "--model_args",
            f"model=served,base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,tokenizer_backend=None,tokenized_requests=False",
            "--tasks",
            "MATH500",
            "--debug",
            "--gen_kwargs",
            "temperature=0.85,top_p=0.91,seed=27",
            "--output_path",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert endpoint.requests
    payload = endpoint.requests[0]
    assert payload["temperature"] == 0.85
    assert payload["seed"] == 27
    assert payload["top_p"] == 0.91
