from pathlib import Path

from lm_eval.tasks import TaskManager

from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR
from eval.generation_stops import truncate_at_stop
from eval.lm_eval_tasks.humaneval.scoring import build_predictions


def test_truncate_at_earliest_turn_boundary():
    output = "answer\nQuestion: repeated prompt<|im_end|>"
    assert truncate_at_stop(output) == "answer"
    assert truncate_at_stop("answer without a boundary") == "answer without a boundary"


def test_math_graders_ignore_boxed_answers_from_a_repeated_turn():
    output = (
        r"Reasoning. \boxed{17}" + "\nQuestion: unrelated\n" + r"Reasoning. \boxed{42}"
    )
    assert AIME24Benchmark.extract_answer(None, output) == "17"
    assert MATH500Benchmark.extract_answer(None, output) == "17"


def test_humaneval_discards_a_closing_code_fence():
    docs = [{"prompt": "def answer():\n"}]
    responses = [["    return 42\n```\nAssistant:\nHere is another answer"]]
    assert build_predictions(responses, docs) == [["def answer():\n    return 42"]]


def test_generation_task_overrides_take_precedence():
    task_manager = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])

    for task_name in ("gsm8k", "humaneval"):
        entry = task_manager.task_index[task_name]
        assert entry.yaml_path is not None
        assert (
            Path(entry.yaml_path)
            .resolve()
            .is_relative_to(Path(DEFAULT_LM_EVAL_INCLUDE_DIR).resolve())
        )

    gsm8k_stops = task_manager.task_index["gsm8k"].cfg["generation_kwargs"]["until"]
    humaneval_stops = task_manager.task_index["humaneval"].cfg["generation_kwargs"][
        "until"
    ]
    assert "\nYou are an AI assistant" in gsm8k_stops
    assert "\nAssistant:" in humaneval_stops
    assert "\n```" in humaneval_stops
