"""Regression coverage for normalized samples written to FineStore."""

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.sample_results import record_sample_metrics
from eval.eval import evaluate, handle_evaluation_output
from eval.eval_tracker import DCEvaluationTracker
from eval.task import BaseBenchmark


class _FakeLM:
    rank = 0
    world_size = 1


class _RecordingBenchmark(BaseBenchmark):
    def __init__(self, generation_result: dict[str, Any], scored_result: dict[str, Any]):
        super().__init__()
        self.generation_result = generation_result
        self.scored_result = scored_result

    def generate_responses(self, model):
        return self.generation_result

    def evaluate_responses(self, results):
        for example in results.get("examples", []):
            record_sample_metrics(example, accuracy=1.0)
        return self.scored_result


class _CustomTaskManager:
    def __init__(self, task_name: str, benchmark: _RecordingBenchmark):
        self.tasks = {task_name: benchmark}
        self.benchmark = benchmark

    def get_benchmark(self, task_name):
        return self.benchmark


class _EmptyPretrainTaskManager:
    all_tasks = {}


def _args(**overrides) -> Namespace:
    values = {
        "model": "local-completions",
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
        "finestore_output_path": None,
        "finestore_output_prefix": "run",
    }
    values.update(overrides)
    return Namespace(**values)


def _write_output(tmp_path: Path, results: dict[str, Any], args: Namespace):
    results["config"] = {"batch_sizes": [1]}
    args.finestore_output_path = str(tmp_path / "archive")
    args.finestore_output_prefix = next(iter(results["results"]))
    tracker = DCEvaluationTracker()
    tracker.general_config_tracker.model_name_sanitized = "test-model"
    handle_evaluation_output(results, args, tracker)
    table = ReadView(args.finestore_output_path).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    return [sample_from_archive_row(row) for row in table.to_pylist(maps_as_pydicts="strict")]


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
def test_custom_scored_tasks_write_canonical_finestore_samples(
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
    samples = _write_output(tmp_path, results, _args())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.task == task_name
    assert sample.metrics == {"accuracy": 1.0}
    assert all(
        key not in json.loads(sample.doc) for key in {"model_output", "model_outputs", "gpt_completion", "response", "output"}
    )


def test_lm_eval_task_uses_same_finestore_sample_contract(tmp_path: Path, monkeypatch):
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
        "metrics": ["exact_match"],
        "exact_match": 1.0,
    }

    def fake_simple_evaluate(*args, **kwargs):
        return {"results": {"gsm8k": {"exact_match": 1.0}}, "samples": {"gsm8k": [native_record]}}

    monkeypatch.setattr("eval.resume.lm_eval_native.resume_simple_evaluate", fake_simple_evaluate)
    args = _args(
        max_batch_size=None,
        device=None,
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

    samples = _write_output(tmp_path, results, args)
    assert len(samples) == 1
    assert samples[0].task == "gsm8k"
    assert samples[0].metrics == {"exact_match": 1.0}


def test_unscored_task_reports_failure():
    benchmark = _RecordingBenchmark(
        {"examples": [{"prompt": "x", "response": "y"}]},
        {"error": "grader failed"},
    )
    result = evaluate(
        lm=_FakeLM(),
        task_manager=_CustomTaskManager("IFEval", benchmark),
        pretrain_task_manager=_EmptyPretrainTaskManager(),
        task_list=["IFEval"],
        task_routes={"IFEval": "Evalchemy chat benchmark"},
        batch_sizes_list=[1],
        args=_args(),
    )
    assert result["results"] == {}
    assert result["task_outcomes"]["IFEval"]["status"] == "failed"
