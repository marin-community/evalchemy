# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for benchmark-independent sample identity and coverage."""

from types import SimpleNamespace

import pytest

from eval.contracts.sample_manifest import SampleCoverageError, SampleIdentityError, SampleManifest, SampleRequest
from eval.resume import ResumeManager, RunFingerprint
from eval.sample_logging import canonicalize_samples
from eval.task import BaseBenchmark


class _Benchmark(BaseBenchmark):
    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError


class _Model:
    rank = 0
    world_size = 1

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def generate_until(self, requests):
        self.calls += 1
        return list(self.outputs)


class _QueuedModel(_Model):
    def generate_until(self, requests):
        self.calls += 1
        return [f"answer-{request.idx}-{request.repeat_idx}" for request in requests]


def _instance(source_id, *, repeat=None):
    instance = SimpleNamespace(
        idx=source_id,
        args=("prompt", {}),
        request_type="generate_until",
        task_name=None,
    )
    if repeat is not None:
        instance.repeat_idx = repeat
    return instance


@pytest.mark.parametrize("source_id", ["sample_0", 7, {"split": "test", "id": [3, "a"]}])
def test_source_id_types_round_trip_without_integer_coercion(source_id):
    manifest = SampleManifest("CruxEval")

    entries = manifest.plan_batch([SampleRequest(source_id=source_id, ordinal=0)])
    manifest.mark_generated(entries, ["answer"])
    manifest.mark_scored(manifest.generated_sample_count)

    assert entries[0].source_id == source_id
    assert manifest.expected_sample_count == 1
    assert manifest.generated_sample_count == 1
    assert manifest.scored_sample_count == 1


@pytest.mark.parametrize("source_id", [None, float("nan"), object()])
def test_invalid_source_identity_fails_before_generation(source_id):
    model = _Model(["unused"])

    with pytest.raises(SampleIdentityError):
        _Benchmark().compute(model, [_instance(source_id)])

    assert model.calls == 0


def test_duplicate_source_identity_fails_before_generation():
    model = _Model(["unused", "unused"])

    with pytest.raises(SampleIdentityError, match="duplicate"):
        _Benchmark().compute(model, [_instance("same"), _instance("same")])

    assert model.calls == 0


def test_source_identity_cannot_move_to_a_different_ordinal():
    manifest = SampleManifest("task")
    manifest.plan_batch([SampleRequest(source_id="stable", ordinal=0)])

    with pytest.raises(SampleIdentityError, match="ordinal"):
        manifest.plan_batch([SampleRequest(source_id="stable", ordinal=1)])


def test_short_model_response_is_a_coverage_failure():
    model = _Model(["only-one"])

    with pytest.raises(SampleCoverageError, match="returned 1 outputs for 2"):
        _Benchmark().compute(model, [_instance("a"), _instance("b")])


def test_namespaces_distinguish_facets_of_the_same_source_dataset():
    benchmark = _Benchmark()
    model = _Model(["answer"])

    benchmark.compute(model, [_instance("sample_0")], sample_namespace="input")
    benchmark.compute(model, [_instance("sample_0")], sample_namespace="output")

    assert benchmark.sample_manifest.expected_sample_count == 2


@pytest.mark.parametrize("source_id", ["sample_0", {"split": "test", "id": [3, "a"]}])
def test_resume_uses_the_shared_opaque_identity(tmp_path, source_id):
    fingerprint = RunFingerprint.from_run_inputs(
        model_repo="model",
        task_name="_",
        task_data_digest="sha256:data",
    )
    first = _Benchmark()
    first.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))
    assert first.compute(_Model(["saved"]), [_instance(source_id)]) == ["saved"]

    resumed_model = _Model([])
    resumed = _Benchmark()
    resumed.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))

    assert resumed.compute(resumed_model, [_instance(source_id)]) == ["saved"]
    assert resumed_model.calls == 0


def test_resume_rejects_changed_identity_before_generation(tmp_path):
    fingerprint = RunFingerprint.from_run_inputs(
        model_repo="model",
        task_name="_",
        task_data_digest="sha256:data",
    )
    first = _Benchmark()
    first.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))
    first.compute(_Model(["saved"]), [_instance("old-id")])

    resumed = _Benchmark()
    resumed.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))
    model = _Model(["unused"])

    with pytest.raises(SampleIdentityError, match="absent from the current request batch"):
        resumed.compute(model, [_instance("changed-id")])

    assert model.calls == 0


def test_sample_logging_uses_manifest_identity_and_coordinates():
    manifest = SampleManifest("task")
    entries = manifest.plan_batch([SampleRequest(source_id="source", ordinal=0, shard=1, repeat=2)])
    manifest.mark_generated(entries, ["answer"])

    records = canonicalize_samples("task", [{"resps": ["answer"]}], manifest)

    assert records[0]["sample_id"] == entries[0].sample_id
    assert records[0]["source_id"] == "source"
    assert records[0]["sample_ordinal"] == 0
    assert records[0]["sample_shard"] == 1
    assert records[0]["sample_repeat"] == 2


def test_sample_logging_coalesces_lm_eval_filter_records_by_document():
    manifest = SampleManifest("gsm8k")
    entries = manifest.plan_batch(
        [SampleRequest(source_id="first", ordinal=0), SampleRequest(source_id="second", ordinal=1)]
    )
    manifest.mark_generated(entries, ["answer-0", "answer-1"])

    def sample(doc_id, filter_name, score):
        return {
            "doc_id": doc_id,
            "doc": {"question": f"question-{doc_id}"},
            "target": f"answer-{doc_id}",
            "arguments": [[f"prompt-{doc_id}", {}]],
            "resps": [[f"response-{doc_id}"]],
            "filtered_resps": [f"filtered-{filter_name}-{doc_id}"],
            "filter": filter_name,
            "metrics": ["exact_match"],
            "exact_match": score,
            "doc_hash": f"doc-{doc_id}",
            "prompt_hash": f"prompt-{doc_id}",
            "target_hash": f"target-{doc_id}",
        }

    records = canonicalize_samples(
        "gsm8k",
        [
            sample(0, "strict-match", 0.0),
            sample(1, "strict-match", 1.0),
            sample(0, "flexible-extract", 1.0),
            sample(1, "flexible-extract", 1.0),
        ],
        manifest,
    )

    assert [record["source_id"] for record in records] == ["first", "second"]
    assert records[0]["filter"] == "strict-match"
    assert records[0]["filter_variants"] == [
        {
            "filter": "strict-match",
            "filtered_resps": ["filtered-strict-match-0"],
            "metrics": {"exact_match": 0.0},
        },
        {
            "filter": "flexible-extract",
            "filtered_resps": ["filtered-flexible-extract-0"],
            "metrics": {"exact_match": 1.0},
        },
    ]


def test_sample_logging_rejects_incomplete_lm_eval_filter_cohorts():
    manifest = SampleManifest("gsm8k")
    entries = manifest.plan_batch(
        [SampleRequest(source_id="first", ordinal=0), SampleRequest(source_id="second", ordinal=1)]
    )
    manifest.mark_generated(entries, ["answer-0", "answer-1"])
    shared = {"doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target"}
    samples = [
        {"doc_id": 0, "filter": "strict-match", **shared},
        {"doc_id": 1, "filter": "strict-match", **shared},
        {"doc_id": 0, "filter": "flexible-extract", **shared},
    ]

    with pytest.raises(SampleCoverageError, match="received 3 records"):
        canonicalize_samples("gsm8k", samples, manifest)


def test_passk_restores_are_included_in_manifest_coverage(tmp_path):
    fingerprint = RunFingerprint.from_run_inputs(
        model_repo="model",
        task_name="_",
        task_data_digest="sha256:data",
    )

    def build_instances(sample_idx, seed):
        del seed
        instances = [_instance("a"), _instance("b")]
        for instance in instances:
            instance.repeat_idx = sample_idx
        return instances

    first = _Benchmark(num_samples=2)
    first.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))
    expected = first.generate_n_samples_batched(
        _QueuedModel([]),
        build_instances,
        batch_size=1,
    )

    resumed = _Benchmark(num_samples=2)
    resumed.attach_resume_manager(ResumeManager(run_dir=tmp_path, fingerprint=fingerprint, mode="auto"))
    resumed_model = _QueuedModel([])

    assert resumed.generate_n_samples_batched(resumed_model, build_instances, batch_size=1) == expected
    resumed.sample_manifest.validate_generated()
    assert resumed.sample_manifest.generated_sample_count == 2
    assert resumed_model.calls == 0
