import sys
from pathlib import Path

import datasets

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.chat_benchmarks.OlympiadBench.eval_instruct import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_SPLIT,
    OlympiadBenchBenchmark,
)


def test_load_questions_uses_full_text_only_english_split(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    rows = [
        {
            "question_id": "text-only",
            "subfield": "Algebra",
            "context": "Use the following lemma.",
            "question": "Find x.",
            "images": [],
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
            "question": "Read the missing diagram.",
            "images": [],
            "final_answer": ["2"],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": "Numerical",
            "error": None,
            "source": "OE_MM_physics_en_COMP",
        },
        {
            "question_id": "proof",
            "subfield": "Geometry",
            "context": None,
            "question": "Prove the claim.",
            "images": [],
            "final_answer": [],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": None,
            "error": None,
            "source": "TP_TO_maths_en_COMP",
        },
    ]
    request = {}

    def load_dataset(name, *, split, revision, cache_dir=None):
        request.update(name=name, split=split, revision=revision, cache_dir=cache_dir)
        return rows

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    questions = OlympiadBenchBenchmark().load_questions()

    assert request == {
        "name": DEFAULT_DATASET,
        "split": DEFAULT_SPLIT,
        "revision": DEFAULT_DATASET_REVISION,
        "cache_dir": None,
    }
    assert questions == [
        {
            "id": "text-only",
            "problem": "Use the following lemma.\n\nFind x.",
            "question": "Find x.",
            "context": "Use the following lemma.",
            "answer": ["1"],
            "subject": "mathematics",
            "subfield": "Algebra",
            "unit": None,
            "answer_type": "Numerical",
            "is_multiple_answer": False,
            "error": None,
            "source": "OE_TO_maths_en_COMP",
        }
    ]
