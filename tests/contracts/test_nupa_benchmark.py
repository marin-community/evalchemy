import json
from pathlib import Path

import pytest

from eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset import convert_file
from eval.chat_benchmarks.NUPA.eval_instruct import (
    BENCHMARK_SIZE,
    NUPABenchmark,
    flatten_nupa_row,
    iter_nupa_source_records,
    split_prompt_answer,
)
from eval.contracts.sample_results import sample_metric_fields


class _AnsweringModel:
    rank = 0
    world_size = 1

    @staticmethod
    def apply_chat_template(messages):
        return messages

    @staticmethod
    def generate_until(instances):
        return [instance.doc["answer"] for instance in instances]


def _source_file(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "add_Integer_Integer_Integer": {
                    "3": [f"Directly return an integer. Add: {value} + 0 = {value}" for value in range(5)],
                    "4": [f"Directly return an integer. Add: {value} + 0 = {value}" for value in range(10, 15)],
                }
            }
        )
    )
    return path


def test_source_loader_reproduces_seeded_per_group_sampling(tmp_path):
    source = _source_file(tmp_path / "test.json")

    records = list(iter_nupa_source_records(source, split="test", num_each=2, random_seed=7))

    assert [(record["digit"], record["answer"]) for record in records] == [
        (3, "2"),
        (3, "1"),
        (4, "13"),
        (4, "10"),
    ]


def test_converter_materializes_the_same_selected_rows(tmp_path):
    source = _source_file(tmp_path / "test.json")
    output = tmp_path / "flattened.jsonl"

    count = convert_file(source, output, split="test", num_each=2, random_seed=7)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 4
    assert [record["answer"] for record in records] == ["2", "1", "13", "10"]


def test_flattening_preserves_metadata_and_uses_short_range_for_digit_21_bug():
    records = flatten_nupa_row(
        {
            "to_float_Fraction_none_Float": {
                "1": ["Convert the number to float: 1/1 = 1.0"],
                "21": ["Convert the number to float: 1/2 = 0.5"],
            }
        },
        split="test",
    )

    assert records == [
        {
            "id": "test:to_float_Fraction_none_Float:1:000000",
            "task_name": "to_float_Fraction_none_Float",
            "operation": "to_float",
            "answer_format": "Float",
            "digit": 1,
            "length_bucket": "S",
            "prompt": "Convert the number to float: 1/1 =",
            "answer": "1.0",
        },
        {
            "id": "test:to_float_Fraction_none_Float:21:000000",
            "task_name": "to_float_Fraction_none_Float",
            "operation": "to_float",
            "answer_format": "Float",
            "digit": 21,
            "length_bucket": "XL",
            "prompt": "Convert the number to float: 1/2 =",
            "answer": "0.5",
        },
    ]


def test_split_prompt_answer_rejects_missing_delimiter():
    with pytest.raises(ValueError, match="no answer delimiter"):
        split_prompt_answer("No answer delimiter")


def test_benchmark_honors_limit_and_reports_reconcilable_scores():
    benchmark = NUPABenchmark(debug=True)
    benchmark.set_evaluation_limits(limit=2)

    generation = benchmark.generate_responses(_AnsweringModel())
    scored = benchmark.evaluate_responses(generation)

    assert len(generation["examples"]) == 2
    assert scored["dataset_num_samples"] == 2
    assert scored["exact_match"] == 1.0
    assert scored["task:max_Float_Float_Float/exact_match"] == 1.0
    assert scored["bucket:S/exact_match"] == 1.0
    assert sample_metric_fields(generation["examples"][0]) == {
        "metrics": ["exact_match", "digit_match", "dlength", "format_valid_rate", "no_answer_rate"],
        "exact_match": 1.0,
        "digit_match": 1.0,
        "dlength": 0.0,
        "format_valid_rate": 1.0,
        "no_answer_rate": 0.0,
    }


def test_benchmark_metadata_describes_the_published_protocol():
    description = NUPABenchmark().describe("NUPA")

    assert description is not None
    assert description.primary_metric == "accuracy"
    assert description.n_benchmark == BENCHMARK_SIZE
    assert description.n_attempted == BENCHMARK_SIZE
