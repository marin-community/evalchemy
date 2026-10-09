"""Contract tests for Evalchemy's FineStore output."""

from argparse import Namespace
from pathlib import Path

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.finestore_output import completed_finestore_output, read_finestore_output, write_finestore_output
from eval.contracts.finestore_resume import FineStoreResumeManager, ResumeRefused
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.contracts.task_outcome import TaskOutcome, TaskRoute, TaskStatus
from eval.eval import cli_evaluate, handle_evaluation_output
from eval.eval_tracker import DCEvaluationTracker
from eval.serve_eval.results import EvalResults
from eval.resume.fingerprint import RunFingerprint
from eval.resume.wiring import build_resume_wiring


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


def test_normalization_marks_transport_failures_as_infrastructure_errors():
    sample = sample_from_lm_eval(
        "FinanceBench",
        {
            "doc_id": 0,
            "doc": {"question": "question"},
            "target": "answer",
            "arguments": [["question", {}]],
            "resps": [[""]],
            "filtered_resps": [""],
            "filter": "none",
            "failure_category": "model_transport",
            "metrics": ["accuracy"],
            "accuracy": 0.0,
        },
    )

    assert sample.output == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
    assert sample.extracted == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
    assert sample.metrics == {"accuracy": 0.0}


def test_normalization_does_not_mark_malformed_model_output_as_infrastructure():
    sample = sample_from_lm_eval(
        "task",
        {
            "doc_id": 0,
            "doc": {},
            "target": "answer",
            "arguments": [["question", {}]],
            "resps": [[""]],
            "filtered_resps": [""],
            "filter": "none",
            "failure_category": "malformed_model_response",
            "metrics": ["accuracy"],
            "accuracy": 0.0,
        },
    )

    assert sample.output == ""
    assert sample.extracted == ""


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


def test_finestore_output_expands_filter_variants(
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

    assert read_finestore_output(root, "gsm8k_5shot") == results
    assert view.read_blob("sources/evalchemy/gsm8k_5shot/native/samples_gsm8k_native.jsonl") is None


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


def test_finestore_resume_restores_committed_units(tmp_path: Path):
    root = str(tmp_path / "archive")
    fingerprint = RunFingerprint(inputs={"task_name": "gsm8k", "model_repo": "test-model"})
    first = FineStoreResumeManager(root, "gsm8k_0shot", "gsm8k", fingerprint)
    assert first.decide() == "fresh"
    first.record({"task": "gsm8k", "problem_idx": 0}, {"output": "42"})
    first.finalize()

    changed = FineStoreResumeManager(root, "gsm8k_0shot", "gsm8k", RunFingerprint(inputs={"task_name": "other"}))
    with pytest.raises(ResumeRefused):
        changed.decide()

    retry = FineStoreResumeManager(root, "gsm8k_0shot", "gsm8k", fingerprint)
    assert retry.decide() == "resume"
    assert retry.restore() == {(('problem_idx', 0), ('task', 'gsm8k')): {"output": "42"}}
    assert retry.should_skip({"task": "gsm8k", "problem_idx": 0})
    retry.record({"task": "gsm8k", "problem_idx": 1}, {"output": "43"})
    retry.finalize()

    assert FineStoreResumeManager(root, "gsm8k_0shot", "gsm8k", fingerprint).restore() == {
        (('problem_idx', 0), ('task', 'gsm8k')): {"output": "42"},
        (('problem_idx', 1), ('task', 'gsm8k')): {"output": "43"},
    }

def test_finestore_resume_refuses_rank_layout_change_and_force_fresh(tmp_path: Path):
    root = str(tmp_path / "archive")
    fingerprint = RunFingerprint(inputs={"task_name": "gsm8k"})
    first = FineStoreResumeManager(root, "run", "gsm8k", fingerprint)
    assert first.decide() == "fresh"
    first.finalize()

    with pytest.raises(ResumeRefused, match="fingerprint changed"):
        FineStoreResumeManager(root, "run", "gsm8k", fingerprint, world_size=2).decide()
    with pytest.raises(ResumeRefused, match="already exists"):
        FineStoreResumeManager(root, "run", "gsm8k", fingerprint, mode="force-fresh").decide()


def test_finestore_output_preserves_completed_task_after_archive_reopens(tmp_path: Path):
    root = str(tmp_path / "archive")
    results = {"results": {"gsm8k": {"exact_match": 0.5}}}
    write_finestore_output(root, "gsm8k_0shot", results, {})
    sealed = ReadView(root)
    assert completed_finestore_output(root, "gsm8k_0shot")
    assert sealed.is_sealed()
    write_finestore_output(root, "gsm8k_0shot", {"results": {"gsm8k": {"exact_match": 0.0}}}, {})
    write_finestore_output(
        root,
        "gsm8k_0shot",
        {"results": {}, "task_outcomes": {"gsm8k": {"status": "failed"}}},
        {},
    )
    assert read_finestore_output(root, "gsm8k_0shot") == results
    assert ReadView(root).token == sealed.token
    other_task = FineStoreResumeManager(root, "arc_0shot", "arc", RunFingerprint(inputs={"task_name": "arc"}))
    other_task.record({"task": "arc", "problem_idx": 0}, {"output": "A"})
    other_task.finalize()
    assert not ReadView(root).is_sealed()
    assert completed_finestore_output(root, "gsm8k_0shot")

def test_failed_finestore_result_can_be_replaced_by_success(tmp_path: Path):
    root = str(tmp_path / "archive")
    failure = {"results": {}, "task_outcomes": {"gsm8k": {"status": "failed"}}}
    success = {"results": {"gsm8k": {"exact_match": 0.5}}}

    write_finestore_output(root, "gsm8k_0shot", failure, {})
    assert not completed_finestore_output(root, "gsm8k_0shot")
    write_finestore_output(root, "gsm8k_0shot", success, {})

    assert completed_finestore_output(root, "gsm8k_0shot")
    assert read_finestore_output(root, "gsm8k_0shot") == success

def test_sample_normalization_failure_does_not_commit_success(tmp_path: Path, monkeypatch):
    root = str(tmp_path / "archive")
    result = {"results": {"gsm8k": {"exact_match": 1.0}}}

    def fail_normalization(task_name, record):
        raise ValueError("invalid sample")

    monkeypatch.setattr("eval.contracts.finestore_output.samples_from_lm_eval", fail_normalization)
    with pytest.raises(ValueError, match="invalid sample"):
        write_finestore_output(root, "gsm8k_0shot", result, {"gsm8k": [{}]})

    assert read_finestore_output(root, "gsm8k_0shot") is None

def test_cli_skips_completed_finestore_result_before_model_setup(tmp_path: Path):
    root = str(tmp_path / "archive")
    result = {"results": {"gsm8k": {"exact_match": 0.5}}}
    write_finestore_output(root, "gsm8k_0shot", result, {})

    cli_evaluate(Namespace(
        finestore_output_path=root,
        finestore_output_prefix="gsm8k_0shot",
        resume_mode="auto",
    ))

    assert read_finestore_output(root, "gsm8k_0shot") == result


def test_finestore_cli_wiring_resumes_without_local_output_path(tmp_path: Path):
    root = str(tmp_path / "archive")
    args = Namespace(
        resume_mode="auto",
        output_path=None,
        finestore_output_path=root,
        finestore_output_prefix="gsm8k_0shot",
        model_args="model=test-model",
    )
    model = Namespace(world_size=1, rank=0)
    unit = {"task": "gsm8k", "problem_idx": 0}
    factory = build_resume_wiring(args, model)
    first = factory("gsm8k")
    first.record(unit, {"output": "42"})
    first.finalize()

    retry = build_resume_wiring(args, model)("gsm8k")
    assert retry.should_skip(unit)
    assert retry.restore()[(('problem_idx', 0), ('task', 'gsm8k'))] == {"output": "42"}


def test_finestore_cli_refuses_resume_off(tmp_path: Path):
    args = Namespace(finestore_output_path=str(tmp_path / "archive"), log_samples=True, resume_mode="off")
    with pytest.raises(ValueError, match="FineStore output requires"):
        cli_evaluate(args)


def test_finestore_mode_writes_no_local_result_files(tmp_path: Path):
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
    tracker = DCEvaluationTracker(None)
    tracker.general_config_tracker.model_name_sanitized = "test-model"

    handle_evaluation_output(results, args, tracker)

    assert list(tmp_path.rglob("samples_*.jsonl")) == []
    assert list(tmp_path.rglob("results_*.json")) == []
    loaded = EvalResults.load_archive(str(archive_root), "gsm8k_0shot")
    assert loaded.results == {"gsm8k": {"exact_match,none": 1.0}}
    assert ReadView(str(archive_root)).read_blob("sources/evalchemy/gsm8k_0shot/native/samples_gsm8k_native.jsonl") is None
