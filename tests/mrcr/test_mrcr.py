import json

import pytest

from eval.chat_benchmarks.MRCR import eval_instruct as mrcr
from eval.contracts.eval_spec import SourceRef
from eval.graders.structured import nonce_similarity
from eval.sources.common import SourceOptions


class _WhitespaceTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()


def _row(needles, suffix, words=4_100):
    nonce = f"nonce{needles}{suffix}"
    return {
        "prompt": json.dumps([
            {"role": "user", "content": "word " * words},
            {"role": "assistant", "content": "distractor"},
            {"role": "user", "content": f"Prepend {nonce} to the requested answer."},
        ]),
        "answer": f"{nonce} target response",
        "random_string_to_prepend": nonce,
        "n_needles": needles,
    }


def test_mrcr_source_pins_data_and_interleaves_selected_cells(monkeypatch):
    rows = [_row(needles, suffix) for needles in (2, 4, 8) for suffix in ("a", "b")]
    calls = []

    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return {"train": iter(rows)}

    monkeypatch.setattr(mrcr, "load_dataset", load_dataset)
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    options = SourceOptions("MRCR", SourceRef(mrcr.DATASET_NAME), context_length=8_192, limit=2)
    specs = list(mrcr.MRCRBenchmark().iter_specs(options))

    assert calls == [(mrcr.DATASET_NAME, {
        "revision": mrcr.DATASET_REVISION,
        "data_files": list(mrcr.DATA_FILES),
        "streaming": True,
        "cache_dir": None,
    })]
    assert [spec.doc["n_needles"] for spec in specs] == [2, 4]
    assert [spec.grader.reference["nonce"] for spec in specs] == ["nonce2a", "nonce4a"]


def test_mrcr_grader_requires_nonce_then_compares_response_body(monkeypatch):
    monkeypatch.setattr(mrcr, "_OFFICIAL_TOKENIZER", _WhitespaceTokenizer())
    source = mrcr.MRCRBenchmark(n_needles=[2])
    monkeypatch.setattr(source, "_load_rows", lambda: iter([_row(2, "a")]))
    spec = next(source.iter_specs(SourceOptions("MRCR", SourceRef("fixture"), context_length=8_192)))

    accepted = nonce_similarity(spec, ["nonce2a target response"])
    rejected = nonce_similarity(spec, ["target response"])

    assert accepted.metrics == {"accuracy": 1.0, "prefix_hit_rate": 1.0}
    assert rejected.metrics == {"accuracy": 0.0, "prefix_hit_rate": 0.0}


@pytest.mark.parametrize(("tokens", "expected"), [(4_095, None), (4_096, 8_192), (8_193, 16_384)])
def test_mrcr_bins_match_published_boundaries(tokens, expected):
    assert mrcr.mrcr_bin(tokens) == expected
