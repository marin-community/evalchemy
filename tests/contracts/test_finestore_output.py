"""Contract tests for Evalchemy's native FineStore output."""

from pathlib import Path

from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.finestore_output import write_finestore_output
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.native_serialization import results_json, samples_jsonl


def test_lm_eval_normalization_maps_multiple_choice_scores():
    sample = sample_from_lm_eval(
        "arc_easy",
        {
            "doc_id": 3,
            "doc": {"choices": {"label": ["A", "B"], "text": ["3", "4"]}},
            "target": "B",
            "arguments": [["2 + 2 =", "3"], ["2 + 2 =", "4"]],
            "resps": [[-2.0, False], [-0.1, True]],
            "filtered_resps": [],
            "filter": "none",
            "metrics": ["acc_norm"],
            "acc_norm": 1.0,
        },
    )

    assert sample.prompt_text == "2 + 2 ="
    assert sample.model_choice == 1
    assert sample.target_choice == 1
    assert [(choice.label, choice.text, choice.loglikelihood) for choice in sample.choices or []] == [
        ("A", "3", -2.0),
        ("B", "4", -0.1),
    ]
    assert sample.correct is True


def test_finestore_output_preserves_jsonl_and_expands_filter_variants(
    tmp_path: Path,
):
    root = str(tmp_path / "archive")
    record = {
        "schema_version": 1,
        "task_name": "gsm8k",
        "doc_id": 7,
        "doc": {"question": "What is 40 + 2?"},
        "target": "42",
        "arguments": [["What is 40 + 2?", {}]],
        "resps": [["The answer is 42"]],
        "filtered_resps": ["42"],
        "filter": "strict-match",
        "metrics": ["exact_match"],
        "filter_variants": [
            {
                "filter": "strict-match",
                "filtered_resps": ["42"],
                "metrics": {"exact_match,strict-match": 0.0},
            },
            {
                "filter": "flexible-extract",
                "filtered_resps": ["42"],
                "metrics": {"exact_match,flexible-extract": 1.0},
            },
        ],
    }

    results = {"results": {"gsm8k": {"exact_match": 1.0}}}
    write_finestore_output(root, "gsm8k_5shot", results, {"gsm8k": [record]})

    view = ReadView(root)
    table = view.scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    rows = table.to_pylist(maps_as_pydicts="strict")
    samples = {row["filter"]: sample_from_archive_row(row) for row in rows}
    assert set(samples) == {"strict-match", "flexible-extract"}
    assert samples["strict-match"].correct is False
    assert samples["flexible-extract"].correct is True

    source = view.read_blob("sources/evalchemy/gsm8k_5shot/native/samples_gsm8k_native.jsonl")
    assert source is not None
    assert source.decode() == samples_jsonl([record])
    aggregate = view.read_blob("sources/evalchemy/gsm8k_5shot/native/results_gsm8k.json")
    assert aggregate is not None
    assert aggregate.decode() == results_json(results)
