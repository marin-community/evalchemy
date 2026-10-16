"""Source positions remain stable across task-set and filesystem ordering."""

import json

from datasets import Dataset
import pytest

from eval.chat_benchmarks.LiveBench import eval_instruct
from eval.chat_benchmarks.LiveBench.livebench import common


def question(category, task, index):
    return {"question_id": f"{category}/{task}/{index}", "turns": [f"Question {index}"],
            "category": category, "task": task, "ground_truth": str(index),
            "livebench_release_date": "2024-08-31", "livebench_removal_date": ""}


@pytest.mark.parametrize("question_source", ["huggingface", "jsonl"])
def test_livebench_orders_namespaces_and_keeps_each_sources_row_order(tmp_path, monkeypatch, question_source):
    source = eval_instruct.LiveBenchBenchmark(question_source=question_source, dataset_name="fixture")
    if question_source == "huggingface":
        rows = {category: [question(category, task, index) for task in ("omega", "alpha") for index in (2, 1)]
                for category in common.LIVE_BENCH_CATEGORIES}
        datasets = {category: Dataset.from_list(values) for category, values in rows.items()}
        monkeypatch.setattr(common, "get_hf_dataset", lambda category: datasets[category])
        expected = [question(category, task, index) for category in sorted(rows)
                    for task in ("alpha", "omega") for index in (2, 1)]
    else:
        source.data_path = str(tmp_path)
        paths, expected = [], []
        for name in ("alpha", "omega"):
            path = tmp_path / "fixture" / name / "question.jsonl"
            path.parent.mkdir(parents=True)
            rows = [question(name, "task", index) for index in (2, 1)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths.append(str(path))
            expected.extend(rows)
        monkeypatch.setattr(eval_instruct.glob, "glob", lambda *_args, **_kwargs: paths[::-1])
    assert list(source.load_questions()) == expected
