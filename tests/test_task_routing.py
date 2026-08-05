"""Regression tests for the task registry boundary in the eval CLI."""

import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from eval.eval import CHAT_BENCHMARK_ROUTE, LM_EVAL_ROUTE, cli_evaluate, resolve_task_routes
from eval.task import TaskManager


def test_task_routing_resolves_lm_eval_and_chat_tasks_without_missing_task_warning(caplog):
    chat_tasks = SimpleNamespace(tasks={"MATH500": object()})
    lm_eval_tasks = SimpleNamespace(all_tasks={"arc_challenge": object()})

    with caplog.at_level(logging.INFO):
        routes = resolve_task_routes(["arc_challenge", "MATH500"], chat_tasks, lm_eval_tasks)

    assert routes == {"arc_challenge": LM_EVAL_ROUTE, "MATH500": CHAT_BENCHMARK_ROUTE}
    assert "arc_challenge -> lm-eval" in caplog.text
    assert "MATH500 -> Evalchemy chat benchmark" in caplog.text
    assert "Task not found" not in caplog.text


def test_task_routing_rejects_unknown_tasks_before_evaluation():
    chat_tasks = SimpleNamespace(tasks={"MATH500": object()})
    lm_eval_tasks = SimpleNamespace(all_tasks={"arc_challenge": object()})

    with pytest.raises(ValueError, match="Unknown evaluation tasks: arc_challeng"):
        resolve_task_routes(["arc_challeng"], chat_tasks, lm_eval_tasks)


def test_cli_rejects_unknown_tasks_before_model_initialization(monkeypatch):
    monkeypatch.setattr("eval.eval.InstructTaskManager", lambda **_kwargs: SimpleNamespace(tasks={}))
    monkeypatch.setattr("eval.eval.PretrainTaskManager", lambda *_args, **_kwargs: SimpleNamespace(all_tasks={}))

    def fail_if_model_initialization_is_reached(*_args, **_kwargs):
        raise AssertionError("model initialization must not run for an unknown task")

    monkeypatch.setattr("eval.eval.initialize_model", fail_if_model_initialization_is_reached)
    args = Namespace(
        config=None,
        batch_size="1",
        output_path=None,
        use_database=False,
        tasks="arc_challeng",
        model_id=None,
        model_name=None,
        model_args="model=served",
        max_length=None,
        max_tokens=None,
        gen_kwargs=None,
        annotator_model="auto",
        debug=False,
        seed=[0, 1234, 1234, 1234],
        system_instruction=None,
        num_samples=1,
        pass_at_k="1,8,32,128",
        verbosity="INFO",
        include_path=None,
    )

    with pytest.raises(ValueError, match="Unknown evaluation tasks: arc_challeng"):
        cli_evaluate(args)


def test_annotator_requirement_query_ignores_non_chat_task_without_warning(monkeypatch, caplog):
    monkeypatch.setattr(TaskManager, "_load_benchmarks", lambda *_args: None)
    task_manager = TaskManager()

    with caplog.at_level(logging.WARNING):
        assert task_manager.requires_annotator_model("arc_challenge") is False

    assert "Task not found" not in caplog.text
