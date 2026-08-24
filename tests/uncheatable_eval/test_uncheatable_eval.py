import math
from pathlib import Path

import datasets
import pytest
from lm_eval.tasks import TaskManager


TASK_CATEGORIES = {
    "uncheatable_eval_wikipedia_english": "wikipedia_english",
    "uncheatable_eval_wikipedia_nonenglish": "wikipedia_nonenglish",
    "uncheatable_eval_github_python": "github_python",
    "uncheatable_eval_github_cpp": "github_cpp",
    "uncheatable_eval_github_javascript": "github_javascript",
    "uncheatable_eval_github_markdown": "github_markdown",
    "uncheatable_eval_github_other": "github_other",
    "uncheatable_eval_bbc_news": "bbc_news",
    "uncheatable_eval_arxiv_physics": "arxiv_physics",
    "uncheatable_eval_arxiv_computer_science": "arxiv_cs",
    "uncheatable_eval_arxiv_math": "arxiv_math",
    "uncheatable_eval_arxiv_other": "arxiv_other",
    "uncheatable_eval_biorxiv_all": "biorxiv_all",
    "uncheatable_eval_ao3_english": "ao3_english",
    "uncheatable_eval_ao3_nonenglish": "ao3_nonenglish",
}


def test_uncheatable_eval_tasks_filter_categories_and_score_bpb(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "content": f"document from {category}",
            "untruncated_content": f"document from {category}",
            "category": category,
            "date": "2026-07-01",
            "url": f"https://example.com/{category}",
            "metadata": "{}",
        }
        for category in TASK_CATEGORIES.values()
    ]
    dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})
    load_calls = []

    def load_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    task_path = Path(__file__).parents[2] / "eval" / "lm_eval_tasks"
    loaded = TaskManager(include_path=task_path, include_defaults=False).load("uncheatable_eval")

    assert set(loaded["tasks"]) == set(TASK_CATEGORIES)
    assert len(load_calls) == len(TASK_CATEGORIES)
    for args, kwargs in load_calls:
        assert args == ()
        assert kwargs["path"] == "Jellyfish042/UncheatableEval-2026-07"
        assert kwargs["revision"] == "65889535d56aa38d448ce7e07b08e6e36c031545"

    for task_name, category in TASK_CATEGORIES.items():
        task = loaded["tasks"][task_name]
        documents = list(task.eval_docs)
        assert [document["category"] for document in documents] == [category]

        document = documents[0]
        request = task.construct_requests(document, ctx="")
        assert request.request_type == "loglikelihood_rolling"
        assert request.args == (document["content"],)

        log_likelihood = -12.0
        result = task.process_results(document, [log_likelihood])
        aggregate_bpb = task.aggregation()["bits_per_byte"]([result["bits_per_byte"]])
        expected_bpb = -log_likelihood / len(document["content"].encode()) / math.log(2)
        assert aggregate_bpb == pytest.approx(expected_bpb)
