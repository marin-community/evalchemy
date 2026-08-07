import math

import datasets
import pytest

from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.chat_benchmarks.OlympiadBenchFull.eval_instruct import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_SPLIT,
    OlympiadBenchFullBenchmark,
)
from eval.task import TaskManager


def test_olympiadbench_aliases_expose_the_legacy_subset_and_full_text_only_set():
    manager = TaskManager(task_list=["OlympiadBench", "OlympiadBenchFull"])

    assert set(manager.tasks) == {"OlympiadBench", "OlympiadBenchFull"}
    assert manager.get_benchmark("OlympiadBench").n_repeat == 10
    assert manager.get_benchmark("OlympiadBenchFull").dataset_revision == DEFAULT_DATASET_REVISION


def test_legacy_olympiadbench_subset_reports_aime_style_repeat_standard_error():
    benchmark = OlympiadBenchBenchmark(n_repeat=2)
    results = benchmark.evaluate_responses(
        {
            "examples": [
                {"answer": ["1"], "model_answers": ["1", "0"]},
                {"answer": ["2"], "model_answers": ["2", "0"]},
            ]
        }
    )

    assert results["num_total"] == 2
    assert results["num_repeat"] == 2
    assert results["accuracy_avg"] == 0.5
    assert results["accuracy_std_err"] == 0.5 / math.sqrt(2)


def test_full_olympiadbench_loads_the_pinned_text_only_dataset(monkeypatch):
    rows = [
        {
            "question_id": "text-only",
            "subfield": "Algebra",
            "context": "Use the lemma.",
            "question": "Find x.",
            "final_answer": ["1"],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": "Numerical",
            "error": None,
            "source": "OE_TO_maths_en_COMP",
        },
        {
            "question_id": "multimodal",
            "subfield": "Mechanics",
            "context": None,
            "question": "Read the diagram.",
            "final_answer": ["2"],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": "Numerical",
            "error": None,
            "source": "OE_MM_physics_en_COMP",
        },
    ]
    request = {}

    def load_dataset(name, *, split, revision, cache_dir=None):
        request.update(name=name, split=split, revision=revision, cache_dir=cache_dir)
        return rows

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    questions = OlympiadBenchFullBenchmark().load_questions()

    assert request == {
        "name": DEFAULT_DATASET,
        "split": DEFAULT_SPLIT,
        "revision": DEFAULT_DATASET_REVISION,
        "cache_dir": None,
    }
    assert [question["id"] for question in questions] == ["text-only"]


def test_full_olympiadbench_reports_dataset_provenance_and_sample_standard_error():
    benchmark = OlympiadBenchFullBenchmark()
    results = benchmark.evaluate_responses(
        {
            "examples": [
                {"answer": ["1"], "model_answer": "1"},
                {"answer": ["2"], "model_answer": "0"},
                {"answer": ["3"], "model_answer": "0"},
            ]
        }
    )

    assert results["dataset_name"] == DEFAULT_DATASET
    assert results["dataset_revision"] == DEFAULT_DATASET_REVISION
    assert results["dataset_split"] == DEFAULT_SPLIT
    assert results["dataset_num_samples"] == 3
    assert results["accuracy"] == 1 / 3
    assert results["accuracy_stderr"] == pytest.approx(1 / 3)
