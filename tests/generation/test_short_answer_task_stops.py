"""Regression coverage for NQ-Open and TriviaQA chat stop policies."""

from collections import Counter
from pathlib import Path
import sys

import pytest
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.filters import build_filter_ensemble
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import robust_api
from eval.completion_response import CompletionClassification, CompletionContentPolicy
from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR
from eval.generation_stops import SHORT_ANSWER_STOP_SEQUENCES, truncate_at_stop

_TASKS = ("nq_open", "triviaqa")


def _task_config(task_name: str) -> dict:
    task_manager = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])
    entry = task_manager.task_index[task_name]
    assert entry.yaml_path is not None
    assert Path(entry.yaml_path).resolve().is_relative_to(
        _REPO_ROOT / "eval" / "lm_eval_tasks"
    ), f"{task_name} did not resolve to its Evalchemy override"
    return load_yaml(entry.yaml_path, resolve_func=True)


def _adapter() -> LocalChatCompletion:
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = CompletionContentPolicy.COMBINE
    adapter.completion_responses = []
    return adapter


def _filtered_response(task_name: str, content: str) -> str:
    config = _task_config(task_name)
    steps = [
        (filter_config["function"], {key: value for key, value in filter_config.items() if key != "function"})
        for filter_config in config["filter_list"][0]["filter"]
    ]
    pipeline = build_filter_ensemble(config["filter_list"][0]["name"], steps)
    instance = Instance("generate_until", {}, (), 0)
    instance.resps = [
        _adapter().parse_generations({"choices": [{"index": 0, "message": {"content": content}}]})[0]
    ]
    pipeline.apply([instance])
    return instance.filtered_resps[config["filter_list"][0]["name"]]


@pytest.mark.parametrize(
    ("task_name", "response", "reference"),
    [
        ("nq_open", "\n\n14 December 1972", "14 December 1972"),
        ("triviaqa", "\n\nSt. John's, Newfoundland", "St Johns Newfoundland"),
    ],
)
def test_short_answer_task_scores_chat_content_after_leading_newlines(task_name, response, reference):
    selected = _filtered_response(task_name, response)

    assert selected == response.strip()
    assert exact_match_hf_evaluate(
        predictions=[selected],
        references=[reference],
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"] == 1.0


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_uses_only_shared_turn_boundaries(task_name):
    config = _task_config(task_name)
    stops = config["generation_kwargs"]["until"]

    assert stops == SHORT_ANSWER_STOP_SEQUENCES
    assert not {"\n", ".", ","}.intersection(stops)
    assert truncate_at_stop("14 December 1972\nQuestion: repeated prompt", stops) == "14 December 1972"


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_rejects_reasoning_only_truncation(task_name):
    _task_config(task_name)
    generated = _adapter().parse_generations(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning_content": "working"},
                }
            ]
        }
    )[0]

    assert generated.response.classification == CompletionClassification.REASONING_ONLY_TRUNCATED
    assert robust_api.completion_response_quality_invalid(Counter([generated.response.classification]))


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_rejects_successful_all_empty_chat_run(task_name):
    _task_config(task_name)
    responses = Counter(
        _adapter()
        .parse_generations(
            {
                "id": f"chatcmpl-{index}",
                "usage": {"completion_tokens": 2},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": None}}],
            }
        )[0]
        .response.classification
        for index in range(3)
    )

    assert robust_api.completion_response_quality_invalid(responses)
