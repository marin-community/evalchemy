import json

import pytest

from eval.chat_benchmarks.MRCR import eval_instruct as mrcr
from eval.limits import MissingContextLengthError


class _WhitespaceTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class _Model:
    rank = 0
    world_size = 1
    tokenizer = _WhitespaceTokenizer()

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, instances):
        return [instance.doc["answer"] for instance in instances]


def _row(needles, suffix, words=4_100):
    nonce = f"nonce{needles}{suffix}"
    messages = [
        {"role": "user", "content": "word " * words},
        {"role": "assistant", "content": "distractor"},
        {"role": "user", "content": f"Prepend {nonce} to the requested answer."},
    ]
    return {
        "prompt": json.dumps(messages),
        "answer": f"{nonce} target response",
        "random_string_to_prepend": nonce,
        "n_needles": needles,
    }


def test_mrcr_requires_the_run_context_contract(monkeypatch):
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    benchmark = mrcr.MRCRBenchmark()

    with pytest.raises(MissingContextLengthError):
        benchmark.generate_responses(_Model())


def test_mrcr_streams_pinned_data_and_honors_balanced_global_limit(monkeypatch):
    rows = [_row(needles, suffix) for needles in (2, 4, 8) for suffix in ("a", "b")]
    load_calls = []

    def load_dataset(path, **kwargs):
        load_calls.append((path, kwargs))
        return {"train": iter(rows)}

    monkeypatch.setattr(mrcr, "load_dataset", load_dataset)
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    benchmark = mrcr.MRCRBenchmark()
    benchmark.set_evaluation_limits(max_length=8_192, max_tokens=256, limit=2)

    generated = benchmark.generate_responses(_Model())
    scored = benchmark.evaluate_responses(generated)

    assert load_calls == [
        (
            mrcr.DATASET_NAME,
            {
                "revision": mrcr.DATASET_REVISION,
                "data_files": list(mrcr.DATA_FILES),
                "streaming": True,
                "cache_dir": None,
            },
        )
    ]
    assert [example["n_needles"] for example in generated["examples"]] == [2, 4]
    assert scored["num_total"] == 2
    assert scored["mrcr_accuracy"] == 1.0
    assert scored["prefix_hit_rate"] == 1.0
    assert scored["mrcr_8192_2needle"] == 1.0
    assert scored["mrcr_8192_4needle"] == 1.0


def test_mrcr_rejects_a_missing_requested_cell(monkeypatch):
    monkeypatch.setattr(mrcr, "load_dataset", lambda *args, **kwargs: {"train": iter([_row(2, "a")])})
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    benchmark = mrcr.MRCRBenchmark()
    benchmark.set_evaluation_limits(max_length=8_192, max_tokens=256, limit=2)

    with pytest.raises(ValueError, match="has no examples for requested cells"):
        benchmark.generate_responses(_Model())


def test_mrcr_keeps_uneven_corrected_dataset_cells(monkeypatch):
    rows = [
        _row(2, "a"),
        _row(2, "b"),
        _row(4, "a"),
        _row(8, "a"),
        _row(8, "b"),
        _row(8, "c"),
    ]
    monkeypatch.setattr(mrcr, "load_dataset", lambda *args, **kwargs: {"train": iter(rows)})
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    benchmark = mrcr.MRCRBenchmark()
    benchmark.set_evaluation_limits(max_length=8_192, max_tokens=256)

    generated = benchmark.generate_responses(_Model())

    assert [example["n_needles"] for example in generated["examples"]] == [2, 4, 8, 2, 8, 8]


@pytest.mark.parametrize(
    ("response", "expected_score", "expected_prefix_hit"),
    [
        ("nonce target response", 1.0, 1.0),
        ("target response", 0.0, 0.0),
        ("nonce different", 5 / 13, 1.0),
    ],
)
def test_mrcr_scoring_requires_nonce_then_compares_body(response, expected_score, expected_prefix_hit):
    score, prefix_hit = mrcr.score_response(response, "nonce target response", "nonce")

    assert score == expected_score
    assert prefix_hit == expected_prefix_hit


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (4_095, None),
        (4_096, 8_192),
        (8_192, 8_192),
        (8_193, 16_384),
        (1_048_576, 1_048_576),
        (1_048_577, None),
    ],
)
def test_mrcr_bins_match_the_published_boundaries(tokens, expected):
    assert mrcr.mrcr_bin(tokens) == expected


def test_mrcr_generated_scores_do_not_change_sample_document_hash(monkeypatch):
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    benchmark = mrcr.MRCRBenchmark()
    example = _row(2, "a")
    example.update({"score": 0.5, "prefix_hit": 1.0, "mrcr_bin_upper": 8_192})
    generated = {"examples": [example]}

    scored = benchmark.evaluate_responses(generated)
    sample = benchmark.to_samples(generated, scored)[0]

    assert "score" not in sample["doc"]
    assert "prefix_hit" not in sample["doc"]
    assert sample["accuracy"] == 0.5
