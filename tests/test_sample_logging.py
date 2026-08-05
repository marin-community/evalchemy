"""Regression coverage for the uniform ``eval --log_samples`` artifact contract."""

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from eval.eval import evaluate, handle_evaluation_output
from eval.eval_tracker import DCEvaluationTracker
from eval.task import BaseBenchmark


class _FakeLM:
    rank = 0
    world_size = 1


class _RecordingBenchmark(BaseBenchmark):
    def __init__(self, generation_result: dict[str, Any], scored_result: dict[str, Any]):
        super().__init__()
        self._generation_result = generation_result
        self._scored_result = scored_result

    def generate_responses(self, model):
        return self._generation_result

    def evaluate_responses(self, results):
        return self._scored_result


class _CustomTaskManager:
    def __init__(self, task_name: str, benchmark: _RecordingBenchmark):
        self.tasks = {task_name: benchmark}
        self._benchmark = benchmark

    def get_list_generate_responses(self, tasks):
        return [self._benchmark.generate_responses for _ in tasks]

    def get_list_evaluates(self, tasks):
        return [self._benchmark.evaluate_responses for _ in tasks]

    def get_benchmark(self, task_name):
        return self._benchmark


class _EmptyPretrainTaskManager:
    all_tasks = {}


def _args(**overrides) -> Namespace:
    values = {
        "model": "local-completions",
        "log_samples": True,
        "model_args": "model=test-model",
        "gen_kwargs": None,
        "num_fewshot": 0,
        "max_tokens": None,
        "use_database": False,
        "debug": False,
        "show_config": False,
        "wandb_args": None,
        "batch_size": "1",
        "limit": None,
        "annotator_model": "auto",
    }
    values.update(overrides)
    return Namespace(**values)


def _write_output(tmp_path: Path, results: dict[str, Any], args: Namespace) -> list[Path]:
    results["config"] = {"batch_sizes": [1]}
    tracker = DCEvaluationTracker(str(tmp_path))
    tracker.general_config_tracker.model_name_sanitized = "test-model"
    handle_evaluation_output(results, args, tracker)
    return list((tmp_path / "test-model").glob("samples_*.jsonl"))


@pytest.mark.parametrize(
    ("task_name", "example"),
    [
        ("MATH500", {"problem": "1 + 1", "answer": "2", "model_output": "\\boxed{2}", "model_answer": "2"}),
        (
            "GPQADiamond",
            {"Question": "Which option?", "answer": "A", "model_outputs": ["A", "B"], "model_answers": ["A", "B"]},
        ),
        ("HumanEvalPlus", {"prompt": "def add(a, b):", "output": "return a + b", "generation": "return a + b"}),
        (
            "MBPPPlus",
            {
                "prompt": "write add",
                "gpt_completion": "```python\\ndef add(a, b): return a+b\\n```",
                "generation": "def add(a, b): return a+b",
            },
        ),
        ("IFEval", {"prompt": "Respond with hello", "response": "hello"}),
    ],
)
def test_log_samples_custom_scored_tasks_write_one_canonical_nonempty_artifact(
    tmp_path: Path, task_name: str, example: dict[str, Any]
):
    generation_result = {"examples": [example]}
    scored_result = {"accuracy": 1.0, "examples": generation_result["examples"]}
    benchmark = _RecordingBenchmark(generation_result, scored_result)

    results = evaluate(
        lm=_FakeLM(),
        task_manager=_CustomTaskManager(task_name, benchmark),
        pretrain_task_manager=_EmptyPretrainTaskManager(),
        task_list=[task_name],
        task_routes={task_name: "Evalchemy chat benchmark"},
        batch_sizes_list=[1],
        args=_args(),
    )

    assert "examples" not in results["results"][task_name]
    artifacts = _write_output(tmp_path, results, _args())
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    record = json.loads(artifacts[0].read_text().strip())
    assert record["schema_version"] == 1
    assert record["task_name"] == task_name
    assert {
        "doc_id",
        "doc",
        "target",
        "arguments",
        "resps",
        "filtered_resps",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    } <= record.keys()
    assert all(
        key not in record["doc"] for key in {"model_output", "model_outputs", "gpt_completion", "response", "output"}
    )


def test_log_samples_lm_eval_native_task_uses_the_same_artifact_contract(tmp_path: Path, monkeypatch):
    native_record = {
        "doc_id": 0,
        "doc": {"question": "1 + 1"},
        "target": "2",
        "arguments": [["1 + 1", {}]],
        "resps": [["2"]],
        "filtered_resps": ["2"],
        "doc_hash": "doc",
        "prompt_hash": "prompt",
        "target_hash": "target",
    }

    def fake_simple_evaluate(*args, **kwargs):
        return {"results": {"gsm8k": {"exact_match": 1.0}}, "samples": {"gsm8k": [native_record]}}

    monkeypatch.setattr("eval.resume.lm_eval_native.resume_simple_evaluate", fake_simple_evaluate)
    args = _args(
        max_batch_size=None,
        device=None,
        use_cache=None,
        check_integrity=False,
        write_out=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        verbosity="INFO",
        predict_only=False,
        confirm_run_unsafe_code=False,
        seed=[0, 1234, 1234, 1234],
    )
    pretrain = type("Pretrain", (), {"all_tasks": {"gsm8k": object()}})()

    results = evaluate(
        lm=_FakeLM(),
        task_manager=type("Custom", (), {"tasks": {}})(),
        pretrain_task_manager=pretrain,
        task_list=["gsm8k"],
        task_routes={"gsm8k": "lm-eval"},
        batch_sizes_list=[1],
        args=args,
    )

    artifacts = _write_output(tmp_path, results, args)
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0
    assert json.loads(artifacts[0].read_text())["task_name"] == "gsm8k"


def test_log_samples_unscored_task_writes_no_placeholder(tmp_path: Path):
    benchmark = _RecordingBenchmark({"examples": [{"prompt": "x", "response": "y"}]}, {"error": "grader failed"})
    results = evaluate(
        lm=_FakeLM(),
        task_manager=_CustomTaskManager("IFEval", benchmark),
        pretrain_task_manager=_EmptyPretrainTaskManager(),
        task_list=["IFEval"],
        task_routes={"IFEval": "Evalchemy chat benchmark"},
        batch_sizes_list=[1],
        args=_args(),
    )

    assert results.get("samples") is None
    assert _write_output(tmp_path, results, _args()) == []
