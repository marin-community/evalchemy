# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for shared grader execution and artifact contracts."""

import json
import os
import sys
import threading
from types import ModuleType

import pytest

from eval.contracts.grading import (
    ArtifactValidationError,
    GenerationArtifactManifest,
    GraderExecutionMode,
    execute_grading_jobs,
    validate_serialized_artifact_manifests,
)
from eval.lm_eval_compat import setup_parser
from eval.task import TaskManager


class _Job:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode


def _worker_pid(_job):
    return os.getpid()


def test_sandboxed_graders_run_outside_driver_threads_and_results_keep_order():
    main_thread = threading.current_thread()
    jobs = [
        _Job("thread-safe", GraderExecutionMode.THREAD_SAFE),
        _Job("sandboxed", GraderExecutionMode.SANDBOXED),
        _Job("serial", GraderExecutionMode.SERIAL),
    ]

    def grade(job):
        if job.mode is not GraderExecutionMode.THREAD_SAFE:
            assert threading.current_thread() is main_thread
        return job.name

    results = execute_grading_jobs(jobs, mode_for=lambda job: job.mode, grade=grade)

    assert results == ["thread-safe", "sandboxed", "serial"]


def test_process_isolated_grader_runs_outside_driver_process():
    job = _Job("isolated", GraderExecutionMode.PROCESS_ISOLATED)

    [worker_pid] = execute_grading_jobs([job], mode_for=lambda item: item.mode, grade=_worker_pid)

    assert worker_pid != os.getpid()


def test_functional_correctness_graders_declare_owned_sandboxes(monkeypatch):
    monkeypatch.setitem(sys.modules, "fire", ModuleType("fire"))
    manager = TaskManager(task_list=["HumanEvalPlus", "MBPPPlus"])

    assert manager.load_failures == {}
    benchmarks = [manager.get_benchmark(name) for name in ("HumanEvalPlus", "MBPPPlus")]
    main_thread = threading.current_thread()

    def grade(benchmark):
        assert threading.current_thread() is main_thread
        return benchmark.benchmark_name

    assert execute_grading_jobs(
        benchmarks,
        mode_for=lambda benchmark: benchmark.GRADER_EXECUTION_MODE,
        grade=grade,
    ) == ["HumanEvalPlus", "MBPPPlus"]

    for name in ("HumanEvalPlus", "MBPPPlus"):
        artifacts = GenerationArtifactManifest.temporary()
        try:
            with pytest.raises(ArtifactValidationError, match="no required artifacts"):
                manager.get_benchmark(name).evaluate_responses({"artifacts": artifacts})
        finally:
            artifacts.cleanup()


def test_functional_correctness_infrastructure_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setitem(sys.modules, "fire", ModuleType("fire"))
    manager = TaskManager(task_list=["HumanEvalPlus", "MBPPPlus"])

    def fail_functional_correctness(**_kwargs):
        raise RuntimeError("os.fork is unsafe")

    artifact_names = {
        "HumanEvalPlus": ("generated-python", "generated_python.jsonl"),
        "MBPPPlus": ("generated-python", "generated_python.jsonl"),
    }
    for name, (artifact_name, filename) in artifact_names.items():
        benchmark = manager.get_benchmark(name)
        monkeypatch.setitem(
            benchmark.evaluate_responses.__func__.__globals__,
            "evaluate_functional_correctness",
            fail_functional_correctness,
        )
        artifacts = GenerationArtifactManifest.temporary()
        artifacts.write_jsonl(artifact_name, filename, [{"task_id": "x"}], expected_count=1)
        try:
            with pytest.raises(RuntimeError, match="os.fork is unsafe"):
                benchmark.evaluate_responses({"artifacts": artifacts})
        finally:
            artifacts.cleanup()


def test_generation_artifact_manifest_writes_and_validates_jsonl_atomically():
    manifest = GenerationArtifactManifest.temporary()

    artifact = manifest.write_jsonl(
        "generated",
        "generated.jsonl",
        [{"id": "a"}, {"id": "b"}],
        expected_count=2,
    )

    assert artifact.path.read_text().count("\n") == 2
    assert not list(artifact.path.parent.glob("*.tmp"))
    assert manifest.to_dict() == {
        "schema_version": 1,
        "artifacts": [
            {
                "name": "generated",
                "path": "generated.jsonl",
                "expected_count": 2,
                "sha256": artifact.sha256,
            }
        ],
    }
    manifest.validate_required()
    validate_serialized_artifact_manifests({"task": manifest.to_dict()})
    manifest.cleanup()
    assert not artifact.path.parent.exists()


@pytest.mark.parametrize("line_separator", ["\u2028", "\u2029"])
def test_generation_artifact_manifest_preserves_unicode_line_separators(line_separator):
    manifest = GenerationArtifactManifest.temporary()
    record = {"completion": f"first{line_separator}second"}

    try:
        artifact = manifest.write_jsonl("generated", "generated.jsonl", [record], expected_count=1)

        assert json.loads(artifact.path.read_text(encoding="utf-8").removesuffix("\n")) == record
        manifest.validate_required()
    finally:
        manifest.cleanup()


def test_serialized_artifact_manifest_rejects_schema_and_digest_corruption():
    manifest = GenerationArtifactManifest.temporary()
    manifest.write_jsonl("generated", "generated.jsonl", [{"id": "a"}], expected_count=1)
    serialized = manifest.to_dict()

    serialized["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        validate_serialized_artifact_manifests({"task": serialized})

    serialized["schema_version"] = 1
    serialized["artifacts"][0]["sha256"] = "not-a-digest".ljust(64, "x")
    with pytest.raises(ValueError, match="sha256"):
        validate_serialized_artifact_manifests({"task": serialized})

    manifest.cleanup()


def test_generation_artifact_manifest_rejects_missing_and_wrong_count():
    manifest = GenerationArtifactManifest.temporary()
    artifact = manifest.write_jsonl("generated", "generated.jsonl", [{"id": "a"}], expected_count=1)
    os.unlink(artifact.path)

    with pytest.raises(ArtifactValidationError, match="missing"):
        manifest.validate_required()

    with pytest.raises(ArtifactValidationError, match="expected 2 records"):
        manifest.write_jsonl("wrong-count", "wrong.jsonl", [{"id": "a"}], expected_count=2)

    manifest.cleanup()


class _CodeCompletionModel:
    rank = 0
    world_size = 1

    def apply_chat_template(self, messages):
        return "\n".join(message["content"] for message in messages)

    def generate_until(self, instances):
        return [f"```python\n# source {instance.doc['task_id']}\nreturn 0\n```" for instance in instances]


@pytest.mark.parametrize("task_name", ["HumanEvalPlus", "MBPPPlus"])
def test_code_generation_limit_preserves_selected_ids_and_grader_artifact(task_name):
    args = setup_parser().parse_args(["--limit", "32"])
    benchmark = TaskManager(task_list=[task_name], limit=args.limit).get_benchmark(task_name)
    if task_name == "HumanEvalPlus":
        source = benchmark.load_examples("python")
    else:
        source = list(benchmark.read_test_examples(os.path.join(benchmark.data_dir, "mbppplus.jsonl")))
        with open(os.path.join(benchmark.data_dir, "mbppplus.jsonl")) as stream:
            original_rows = [json.loads(line) for line in stream]
        demonstration_ids = {row["task_id"] for row in original_rows[1:4]}
        assert {row["task_id"] for row in source}.isdisjoint(demonstration_ids)
    selected = source[:32]

    results = benchmark.generate_responses(_CodeCompletionModel())
    artifacts = results["artifacts"]
    try:
        artifacts.validate_required()
        records = [json.loads(line) for line in artifacts.path("generated-python").read_text().splitlines()]
        assert len(records) == 32
        assert [record["task_id"] for record in records] == [record["task_id"] for record in selected]
        if task_name == "MBPPPlus":
            assert {record["task_id"] for record in records}.isdisjoint(demonstration_ids)
        assert [record["prompt"] for record in records] == [record["prompt"] for record in selected]
        for record, original in zip(records, selected, strict=True):
            assert {key: record[key] for key in original} == original
        output_key = "output" if task_name == "HumanEvalPlus" else "gpt_completion"
        for record in records:
            assert f"# source {record['task_id']}" in record[output_key]
            assert f"# source {record['task_id']}" in record["generation"]
        scored = benchmark.evaluate_responses(results)
        assert scored["scored_count"] == 32
    finally:
        artifacts.cleanup()


@pytest.mark.parametrize("task_name", ["HumanEvalPlus", "MBPPPlus"])
def test_code_fractional_limit_uses_native_sample_selection(task_name):
    args = setup_parser().parse_args(["--limit", "0.1"])
    benchmark = TaskManager(task_list=[task_name], limit=args.limit).get_benchmark(task_name)
    source_size = benchmark.benchmark_size()
    expected_size = (source_size + 9) // 10
    assert benchmark.evaluation_limit == expected_size
    assert len(benchmark.limit_samples(range(source_size))) == expected_size


def test_humaneval_fractional_limit_uses_each_language_collection(tmp_path):
    original = TaskManager(task_list=["HumanEvalPlus"]).get_benchmark("HumanEvalPlus")
    examples = original.load_examples("python")
    language_rows = {"python": examples[:20], "cpp": examples[20:30]}
    for language, rows in language_rows.items():
        (tmp_path / f"humanevalplus-{language}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    benchmark = TaskManager(
        task_list=["HumanEvalPlus"], languages=["python", "cpp"], data_dir=str(tmp_path), limit=0.1
    ).get_benchmark("HumanEvalPlus")
    results = benchmark.generate_responses(_CodeCompletionModel())
    artifacts = results["artifacts"]
    try:
        for language, expected_count in (("python", 2), ("cpp", 1)):
            records = [json.loads(line) for line in artifacts.path(f"generated-{language}").read_text().splitlines()]
            assert [row["task_id"] for row in records] == [
                row["task_id"] for row in language_rows[language][:expected_count]
            ]
    finally:
        artifacts.cleanup()
