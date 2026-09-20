"""Contract tests for Evalchemy's native FineStore output."""

from argparse import Namespace
from pathlib import Path

from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.finestore_output import write_finestore_output
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.contracts.task_outcome import TaskOutcome, TaskRoute, TaskStatus
from eval.eval import handle_evaluation_output
from eval.eval_tracker import DCEvaluationTracker
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


def test_normalized_metrics_exclude_sample_identity_coordinates():
    sample = sample_from_lm_eval(
        "MATH500",
        {
            "schema_version": 1,
            "task_name": "MATH500",
            "doc_id": 0,
            "doc": {"problem": "1 + 1"},
            "target": "2",
            "arguments": [["1 + 1", {}]],
            "resps": [["2"]],
            "filtered_resps": ["2"],
            "filter": "none",
            "doc_hash": "doc",
            "prompt_hash": "prompt",
            "target_hash": "target",
            "sample_id": "abc",
            "source_id": 3,
            "sample_ordinal": 7,
            "sample_namespace": "MATH500",
            "sample_shard": None,
            "sample_repeat": 2,
            "metrics": ["accuracy"],
            "accuracy": 1.0,
        },
    )

    assert sample.metrics == {"accuracy": 1.0}
    assert sample.grading is not None
    assert (sample.grading.metric, sample.grading.score) == ("accuracy", 1.0)


def test_finestore_output_preserves_repeated_trials(tmp_path: Path):
    root = str(tmp_path / "archive")
    records = [
        {
            "doc_id": 7,
            "doc": {"question": "Choose A"},
            "target": "A",
            "arguments": [["Choose A", {}]],
            "resps": [[answer]],
            "filtered_resps": [answer],
            "filter": "none",
            "sample_repeat": repeat,
            "metrics": ["accuracy"],
            "accuracy": score,
        }
        for repeat, (answer, score) in enumerate((("A", 1.0), ("B", 0.0), ("A", 1.0)))
    ]

    write_finestore_output(
        root,
        "gpqa_diamond",
        {"results": {"GPQADiamond": {"accuracy_avg": 2 / 3}}},
        {"GPQADiamond": records},
    )

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    rows = sorted(table.to_pylist(maps_as_pydicts="strict"), key=lambda row: row["trial_id"])
    assert [row["trial_id"] for row in rows] == ["0", "1", "2"]
    assert [sample_from_archive_row(row).correct for row in rows] == [True, False, True]


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


def test_finestore_output_keeps_repeated_task_configurations_distinct(tmp_path: Path):
    root = str(tmp_path / "archive")
    record = {
        "doc_id": 7,
        "doc": {"query": "Complete the sentence"},
        "target": "answer",
        "arguments": [["Complete the sentence"]],
        "resps": [["answer"]],
        "filtered_resps": ["answer"],
        "filter": "none",
        "acc,none": 1.0,
    }
    results = {"results": {"hellaswag": {"acc,none": 1.0}}}

    write_finestore_output(root, "hellaswag_0shot", results, {"hellaswag": [record]})
    write_finestore_output(root, "hellaswag_10shot", results, {"hellaswag": [record]})

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    assert set(table.column("task").to_pylist()) == {"hellaswag_0shot", "hellaswag_10shot"}


def test_finestore_mode_does_not_write_a_second_sample_jsonl(tmp_path: Path):
    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    record = {
        "doc_id": 0,
        "doc": {"question": "2 + 2?"},
        "target": "4",
        "arguments": [["2 + 2?"]],
        "resps": [["4"]],
        "filtered_resps": ["4"],
        "exact_match,none": 1.0,
        "doc_hash": "doc",
        "prompt_hash": "prompt",
        "target_hash": "target",
    }
    outcome = TaskOutcome(
        task_name="gsm8k",
        route=TaskRoute.LM_EVAL,
        status=TaskStatus.SUCCEEDED,
        metrics={"exact_match,none": 1.0},
        expected_count=1,
        generated_count=1,
        scored_count=1,
    )
    results = {
        "results": {"gsm8k": {"exact_match,none": 1.0}},
        "samples": {"gsm8k": [record]},
        "task_outcomes": {"gsm8k": outcome.to_dict()},
        "config": {"batch_sizes": [1]},
    }
    args = Namespace(
        log_samples=True,
        show_config=False,
        wandb_args=None,
        use_database=False,
        debug=False,
        model="local-completions",
        model_id=None,
        model_name=None,
        creation_location=None,
        created_by=None,
        is_external=False,
        model_args="model=test",
        gen_kwargs=None,
        num_fewshot=0,
        max_length=None,
        max_tokens=None,
        limit=None,
        annotator_model="auto",
        batch_size="1",
        finestore_output_path=str(archive_root),
        finestore_output_prefix="gsm8k_0shot",
    )
    tracker = DCEvaluationTracker(str(local_root))
    tracker.general_config_tracker.model_name_sanitized = "test-model"

    handle_evaluation_output(results, args, tracker)

    assert list(local_root.rglob("samples_*.jsonl")) == []
    source = ReadView(str(archive_root)).read_blob("sources/evalchemy/gsm8k_0shot/native/samples_gsm8k_native.jsonl")
    assert source == samples_jsonl([record]).encode()
