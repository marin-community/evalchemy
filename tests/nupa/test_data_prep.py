import json

import pytest

from eval.chat_benchmarks.NUPA.data_prep.flatten_hf_dataset import _provenance, convert_file


def test_convert_file_streams_original_schema_and_limits_each_group(tmp_path):
    source = tmp_path / "test.json"
    source.write_text(
        json.dumps(
            {
                "add_Integer_Integer_Integer": {
                    "3": [
                        "Directly return an integer. Add: 830 + 70 = 900",
                        "Directly return an integer. Add: 98 + 150 = 248",
                    ],
                    "4": ["Directly return an integer. Add: 1000 + 1 = 1001"],
                }
            }
        )
    )
    output = tmp_path / "flattened.jsonl"

    count = convert_file(source, output, split="test", limit_per_task_digit=1)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 2
    assert [(record["digit"], record["answer"]) for record in records] == [(3, "900"), (4, "1001")]
    assert all(record["task_name"] == "add_Integer_Integer_Integer" for record in records)


def test_convert_file_rejects_nonpositive_limit(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        convert_file(tmp_path / "unused.json", tmp_path / "unused.jsonl", split="test", limit_per_task_digit=0)


def test_dataset_card_records_schema_source_revision_and_reproduction_command():
    card = _provenance("HaotongYang/NUPA_text", "source-sha", "default", "test")
    normalized_card = " ".join(card.split())

    assert "Source revision: `source-sha`" in card
    assert "`task_name`: original NUPA task-family" in card
    assert "--revision source-sha" in card
    assert "--limit-per-task-digit" in card
    assert "Do not use a limited conversion to report benchmark performance" in normalized_card
