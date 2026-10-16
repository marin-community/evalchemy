"""Emit custom benchmark samples when their graders record per-example metrics."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol

from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.grading import generation_artifacts
from eval.contracts.lm_eval_normalization import is_infrastructure_failure
from eval.contracts.sample_manifest import (
    DEFAULT_SAMPLE_GROUP,
    DEFAULT_SAMPLE_NAMESPACE,
    SampleCoverageError,
    SampleEntry,
    SampleManifest,
)
from eval.contracts.sample_results import (
    SAMPLE_COORDINATES_ANNOTATION,
    SAMPLE_FAILURES_ANNOTATION,
    emit_scored_samples,
    validate_sample_metrics,
)
from eval.sample_logging import canonicalize_sample


class CustomBenchmark(Protocol):
    """Grading, reduction, and sample adapters required by the custom-task scorer."""

    @property
    def sample_manifest(self) -> SampleManifest: ...

    def grade_responses(self, results: dict[str, Any]) -> dict[str, Any]: ...

    def aggregate_results(self, results: dict[str, Any]) -> dict[str, Any]: ...

    def sample_groups(self, results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]: ...

    def with_sample_groups(
        self, results: dict[str, Any], groups: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]: ...

    def to_sample_group(self, namespace: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ScoredResponses:
    """Task metrics and the complete grading evidence from which they were reduced."""

    graded_result: dict[str, Any]
    metrics: dict[str, Any]
    excluded_count: int


def score_responses(
    task_name: str,
    benchmark: CustomBenchmark,
    generation_result: dict[str, Any],
    output_sink: FineStoreOutput | None = None,
) -> ScoredResponses:
    """Grade responses, persist their evidence, and reduce the valid graded subset."""
    artifacts = generation_artifacts(generation_result)
    if artifacts is not None:
        artifacts.validate_required()
    emitter = CustomSampleEmitter(task_name, benchmark, generation_result, output_sink)
    with emit_scored_samples(emitter.emit):
        graded = benchmark.grade_responses(generation_result)
    return _reduced_responses(benchmark, graded, emitter)


def aggregate_responses(
    task_name: str,
    benchmark: CustomBenchmark,
    graded_result: dict[str, Any],
    output_sink: FineStoreOutput | None = None,
) -> ScoredResponses:
    """Persist and reduce restored grading evidence without invoking the grader."""
    emitter = CustomSampleEmitter(task_name, benchmark, graded_result, output_sink)
    for examples in benchmark.sample_groups(graded_result).values():
        for example in examples:
            emitter.emit(example)
    return _reduced_responses(benchmark, graded_result, emitter)


def _reduced_responses(
    benchmark: CustomBenchmark, graded: dict[str, Any], emitter: CustomSampleEmitter
) -> ScoredResponses:
    emitter.finish()
    ordered = emitter.ordered_results(graded)
    metrics = benchmark.aggregate_results(emitter.valid_results(ordered))
    emitter.validate_partial_coverage()
    return ScoredResponses(ordered, metrics, emitter.excluded_count)


class SampleEmissionError(RuntimeError):
    """A custom grader could not serialize a scored sample."""


class CustomSampleEmitter:
    """Bind generated examples to stable identities and append each completed score."""

    def __init__(
        self,
        task_name: str,
        benchmark: CustomBenchmark,
        generation_result: Mapping[str, Any],
        output_sink: FineStoreOutput | None,
    ) -> None:
        self.task_name = task_name
        self.benchmark = benchmark
        self.output_sink = output_sink
        self._examples: set[tuple[str, int, int]] = set()
        self._seen: set[tuple[str, int, int]] = set()
        self._infrastructure_repeats: dict[tuple[str, int, int], set[int | None]] = {}
        self._entries: dict[tuple[str, int, int | None], SampleEntry] = {}
        self._sample_entries: dict[tuple[str, int], SampleEntry] = {}
        self._unit_repeats: dict[tuple[str, int], set[int | None]] = {}
        self._repeat_shape: tuple[int | None, ...] | None = None

        groups = benchmark.sample_groups(generation_result)
        self._group_names = tuple(groups)
        expected_coordinates = set()
        position = 0
        for name, examples in groups.items():
            for ordinal, example in enumerate(examples):
                expected_coordinates.add((name, ordinal, position))
                coordinates = tuple(example.get(SAMPLE_COORDINATES_ANNOTATION, (name, ordinal, position)))
                example[SAMPLE_COORDINATES_ANNOTATION] = coordinates
                self._examples.add(coordinates)
                position += 1
        if self._examples != expected_coordinates:
            raise SampleCoverageError(f"{task_name}: grading evidence has invalid source coordinates")

        manifest = benchmark.sample_manifest
        if manifest.expected_sample_count:
            for entry in manifest.unit_entries():
                self._entries[(entry.namespace, entry.ordinal, entry.repeat)] = entry
                self._sample_entries.setdefault((entry.namespace, entry.ordinal), entry)
                self._unit_repeats.setdefault((entry.namespace, entry.ordinal), set()).add(entry.repeat)
            if position != manifest.expected_sample_count:
                raise SampleCoverageError(
                    f"{task_name}: sample emitter received {position} examples but the manifest "
                    f"contains {manifest.expected_sample_count} samples"
                )

    def emit(self, example: MutableMapping[str, Any]) -> None:
        """Serialize and append the example that just received its score."""
        try:
            coordinates = tuple(example[SAMPLE_COORDINATES_ANNOTATION])
            name, ordinal, position = coordinates
            if coordinates not in self._examples:
                raise SampleCoverageError(f"{self.task_name}: unknown scored sample {coordinates}")
            if coordinates in self._seen:
                raise SampleCoverageError(f"{self.task_name}: sample {position} was scored more than once")
            records = self.benchmark.to_sample_group(name, [example])
            if not records:
                raise SampleCoverageError(f"{self.task_name}: scored sample {position} has no sample record")
            repeat_shape = tuple(record.get("sample_repeat") for record in records)
            namespace = DEFAULT_SAMPLE_NAMESPACE if name == DEFAULT_SAMPLE_GROUP else name
            self._validate_repeats(namespace, ordinal, position, repeat_shape)
            failures = example.get(SAMPLE_FAILURES_ANNOTATION)
            if failures is not None and len(failures) != len(records):
                raise SampleCoverageError(f"{self.task_name}: grading evidence has incomplete failure classifications")
            categories = []
            for index, sample in enumerate(records):
                if failures is not None and failures[index] is not None:
                    sample["failure_category"] = failures[index]
                sample["doc_id"] = position
                entry = self._sample_entry(namespace, ordinal, position, sample.get("sample_repeat"))
                record = canonicalize_sample(self.task_name, sample, position, entry)
                validate_sample_metrics(self.task_name, [record])
                categories.append(record.get("failure_category"))
                if is_infrastructure_failure(record):
                    self._infrastructure_repeats.setdefault(coordinates, set()).add(sample.get("sample_repeat"))
                if self.output_sink is not None:
                    self.output_sink.write_sample(self.task_name, record)
            self._seen.add(coordinates)
            example[SAMPLE_FAILURES_ANNOTATION] = categories
        except Exception as exc:
            raise SampleEmissionError(f"{self.task_name}: failed to emit scored sample: {exc}") from exc

    def _validate_repeats(
        self, namespace: str, ordinal: int, position: int, repeat_shape: tuple[int | None, ...]
    ) -> None:
        if self._repeat_shape is None:
            self._repeat_shape = repeat_shape
        elif repeat_shape != self._repeat_shape:
            raise SampleCoverageError(f"{self.task_name}: examples have different repetition counts")
        if repeat_shape != (None,) and self._entries:
            expected_repeats = self._unit_repeats.get((namespace, ordinal), set())
            if set(repeat_shape) != expected_repeats:
                raise SampleCoverageError(f"{self.task_name}: scored sample {position} has incomplete repeated trials")

    def _sample_entry(self, namespace: str, ordinal: int, position: int, repeat: int | None) -> SampleEntry | None:
        entry = self._entries.get((namespace, ordinal, repeat))
        if entry is None and repeat is None:
            entry = self._sample_entries.get((namespace, ordinal))
        if self._entries and entry is None:
            raise SampleCoverageError(f"{self.task_name}: scored sample {position} has no manifest identity")
        return entry

    def finish(self) -> None:
        """Verify every generated example was emitted and make its rows readable."""
        if len(self._seen) != len(self._examples):
            raise SampleCoverageError(
                f"{self.task_name}: emitted {len(self._seen)} of {len(self._examples)} scored examples"
            )
        try:
            if self.output_sink is not None:
                self.output_sink.finish_task()
        except Exception as exc:
            raise SampleEmissionError(f"{self.task_name}: failed to flush scored samples: {exc}") from exc

    @property
    def excluded_count(self) -> int:
        """Number of serialized trials excluded because of infrastructure failures."""
        return sum(len(repeats) for repeats in self._infrastructure_repeats.values())

    def ordered_results(self, graded_result: dict[str, Any]) -> dict[str, Any]:
        """Require complete returned grading evidence and restore source-coordinate order."""
        groups = self.benchmark.sample_groups(graded_result)
        coordinates = [
            tuple(example[SAMPLE_COORDINATES_ANNOTATION])
            for examples in groups.values() for example in examples
        ]
        if len(coordinates) != len(self._examples) or set(coordinates) != self._examples:
            raise SampleCoverageError(f"{self.task_name}: returned grading evidence differs from the generated samples")
        for namespace, examples in groups.items():
            if any(example[SAMPLE_COORDINATES_ANNOTATION][0] != namespace for example in examples):
                raise SampleCoverageError(f"{self.task_name}: grader changed a sample's namespace")
        ordered = {
            name: sorted(groups.get(name, []), key=lambda example: example[SAMPLE_COORDINATES_ANNOTATION][1])
            for name in self._group_names
        }
        return self.benchmark.with_sample_groups(graded_result, ordered)

    def validate_partial_coverage(self) -> None:
        """Reject partial repeated trials whose coverage cannot be expressed as source counts."""
        if self._infrastructure_repeats and self._repeat_shape != (None,):
            raise SampleCoverageError(f"{self.task_name}: partial repeated samples are not supported by source coverage")

    def valid_results(self, graded_result: dict[str, Any]) -> dict[str, Any]:
        """Exclude failed single samples or annotate repeated examples with their valid trial indices."""
        if not self._infrastructure_repeats:
            return graded_result
        groups = self.benchmark.sample_groups(graded_result)
        valid = {
            name: [
                {
                    **example,
                    "valid_repeats": [
                        repeat for repeat in self._repeat_shape
                        if repeat not in self._infrastructure_repeats.get(tuple(example[SAMPLE_COORDINATES_ANNOTATION]), set())
                    ],
                }
                if self._repeat_shape != (None,) else example
                for example in examples
                if None not in self._infrastructure_repeats.get(tuple(example[SAMPLE_COORDINATES_ANNOTATION]), set())
            ]
            for name, examples in groups.items()
        }
        return {
            **self.benchmark.with_sample_groups(graded_result, valid),
            "excluded_examples": [
                example for examples in groups.values() for example in examples
                if tuple(example[SAMPLE_COORDINATES_ANNOTATION]) in self._infrastructure_repeats
            ],
        }
