"""Generation settings observed at an OpenAI-compatible endpoint."""

import base64
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers
import yaml
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string
from tokenizers import Tokenizer, models, pre_tokenizers

from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.GPQADiamond import eval_instruct as gpqa_module
from eval.chat_benchmarks.GPQADiamond.eval_instruct import GPQADiamondBenchmark
from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.contracts.sample_manifest import SampleManifest
from eval.resume.lm_eval_native import resume_simple_evaluate
from eval.robust_api import configure_generation_overrides
from eval.serve_eval.providers import ServedModel
from eval.serve_eval.run import LOCAL_CHAT_COMPLETIONS, LOCAL_COMPLETIONS, build_model_args


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


@pytest.mark.parametrize(
    "checkpoint",
    ["moonshotai/Kimi-Linear-48B-A3B-Instruct", "google/gemma-4-26B-A4B-it"],
)
def test_served_chat_checkpoint_generates_without_client_tokenizer(endpoint, checkpoint, monkeypatch):
    def reject_local_tokenizer(*_args, **_kwargs):
        raise AssertionError("chat evaluation must use the server's tokenizer")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", reject_local_tokenizer)
    served = ServedModel(
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1",
        model=checkpoint,
        tokenizer=checkpoint,
    )
    model = LocalChatCompletion(**simple_parse_args_string(build_model_args(served, LOCAL_CHAT_COMPLETIONS)))
    request = Instance("generate_until", {"question": "Question"}, ([{"role": "user", "content": "Question"}], {}), 0)

    assert model.generate_until([request]) == ["\\boxed{42}"]
    assert endpoint.requests[0]["model"] == checkpoint
    assert endpoint.requests[0]["messages"] == [{"role": "user", "content": "Question"}]


def test_served_chat_forwards_template_kwargs_and_extra_body(endpoint):
    template_kwargs = {"enable_thinking": False, "add_generation_prompt": True}
    encoded_kwargs = base64.urlsafe_b64encode(json.dumps(template_kwargs).encode()).decode()
    model_args = simple_parse_args_string(
        "model=served,"
        f"base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,"
        "tokenizer_backend=None,tokenized_requests=False,"
        f"chat_template_kwargs=base64:{encoded_kwargs},extra_body={{\"priority\":7}}"
    )
    model = LocalChatCompletion(**model_args)
    request = Instance("generate_until", {"question": "Question"}, ([{"role": "user", "content": "Question"}], {}), 0)

    assert model.generate_until([request]) == ["\\boxed{42}"]
    assert endpoint.requests[0]["chat_template_kwargs"] == template_kwargs
    assert endpoint.requests[0]["priority"] == 7
    assert "extra_body" not in endpoint.requests[0]


def test_served_completions_loads_custom_tokenizer_with_remote_code(monkeypatch):
    def load_custom_tokenizer(_checkpoint, *, trust_remote_code, **_kwargs):
        if not trust_remote_code:
            raise ValueError("custom tokenizer requires trust_remote_code=True")

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, _text, **_kwargs):
                return SimpleNamespace(input_ids=[1, 2])

        return Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_custom_tokenizer)
    served = ServedModel(base_url="http://127.0.0.1:8000/v1", model="custom/checkpoint", tokenizer="custom/checkpoint")
    model = LocalCompletionsAPI(**simple_parse_args_string(build_model_args(served, LOCAL_COMPLETIONS)))

    assert model.tok_encode("Hello") == [1, 2]


def test_served_completions_loads_transformers5_tokenizer_config(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "<|video|>": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "unk_token": "[UNK]",
                "extra_special_tokens": ["<|video|>"],
            }
        )
    )

    model = LocalCompletionsAPI(
        model="served",
        tokenizer=str(tmp_path),
        base_url="http://127.0.0.1:8000/v1/completions",
        tokenizer_backend="huggingface",
    )

    assert model.tok_encode("hello") == [1]
    assert model.tok_encode("<|video|>") == [2]


def _assert_temperature_and_seed(payload, expected):
    for key in ("temperature", "seed"):
        if key in expected:
            assert payload[key] == expected[key]
        else:
            assert key not in payload


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
    _assert_temperature_and_seed(payload, expected)
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
    _assert_temperature_and_seed(payload, overrides)
    if "top_p" in overrides:
        assert payload["top_p"] == overrides["top_p"]
    else:
        assert "top_p" not in payload


def _assert_seeded_requests(requests, first_seed, count):
    assert [request["seed"] for request in requests] == list(range(first_seed, first_seed + count))
    assert all(request["temperature"] == 0.7 for request in requests)


def test_aime24_repeats_use_distinct_sampling_seeds(endpoint, tmp_path):
    data_file = tmp_path / "aime24.jsonl"
    data_file.write_text(json.dumps({"id": "one", "problem": "What is 6 times 7?", "expected_answer": "42"}) + "\n")
    benchmark = AIME24Benchmark(data_file=str(data_file), seed=[42, 42, 42, 42])

    result = benchmark.generate_responses(_model(endpoint, LocalChatCompletion))

    assert len(result["examples"][0]["model_outputs"]) == 10
    _assert_seeded_requests(endpoint.requests, 42, 10)


def test_gpqa_repeats_use_distinct_sampling_seeds(endpoint, monkeypatch):
    dataset = {
        "train": [
            {
                "Question": "What is 6 times 7?",
                "Correct Answer": "42",
                "Incorrect Answer 1": "40",
                "Incorrect Answer 2": "41",
                "Incorrect Answer 3": "43",
            }
        ]
    }
    monkeypatch.setattr(gpqa_module, "load_dataset", lambda *_args, **_kwargs: dataset)
    benchmark = GPQADiamondBenchmark(seed=[42, 42, 42, 42])

    result = benchmark.generate_responses(_model(endpoint, LocalChatCompletion))

    assert len(result["examples"][0]["model_outputs"]) == 3
    _assert_seeded_requests(endpoint.requests, 42, 3)


def test_olympiadbench_repeats_use_distinct_sampling_seeds(endpoint, tmp_path):
    data_file = tmp_path / "olympiadbench.jsonl"
    data_file.write_text(json.dumps({"problem": "What is 6 times 7?", "answer": ["42"]}) + "\n")
    benchmark = OlympiadBenchBenchmark(
        data_file=str(data_file),
        seed=[42, 42, 42, 42],
        judge_api_key="test-key",
    )

    result = benchmark.generate_responses(_model(endpoint, LocalChatCompletion))

    assert len(result["examples"][0]["model_outputs"]) == 10
    _assert_seeded_requests(endpoint.requests, 42, 10)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"temperature": 0.85, "seed": 27, "top_p": 0.91}, {"max_gen_toks": 16000}],
)
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
    _assert_temperature_and_seed(payload, overrides)
    assert payload["top_p"] == overrides.get("top_p", 0.4)
    if "max_gen_toks" in overrides:
        assert payload["max_tokens"] == overrides["max_gen_toks"]
        assert "max_gen_toks" not in payload


@pytest.mark.parametrize("task", ["AIME24", "MATH500"])
def test_cli_output_cap_wins_over_math_benchmark_defaults(endpoint, tmp_path, task):
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
            task,
            "--debug",
            "--limit",
            "1",
            "--max_length",
            "65536",
            "--max_tokens",
            "16000",
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
    for payload in endpoint.requests:
        assert payload["max_tokens"] == 16000
        assert "max_gen_toks" not in payload
        assert payload["temperature"] == 0.85
        assert payload["seed"] == 27
        assert payload["top_p"] == 0.91


def test_cli_generation_limits_win_over_config_defaults(endpoint, tmp_path):
    root = Path(__file__).parents[2]
    config = tmp_path / "tasks.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "tasks": [{"task_name": "AIME24", "batch_size": "auto"}],
                "max_length": 32768,
                "max_tokens": 8192,
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.eval",
            "--model",
            "local-chat-completions",
            "--model_args",
            f"model=served,base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,tokenizer_backend=None,tokenized_requests=False,max_length=65536",
            "--config",
            str(config),
            "--debug",
            "--limit",
            "1",
            "--gen_kwargs",
            "max_gen_toks=16000",
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
    assert all(payload["max_tokens"] == 16000 for payload in endpoint.requests)
    assert all("max_gen_toks" not in payload for payload in endpoint.requests)
