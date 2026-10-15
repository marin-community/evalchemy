"""Emit custom benchmark samples when their graders record per-example metrics."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.sample_manifest import DEFAULT_SAMPLE_GROUP, DEFAULT_SAMPLE_NAMESPACE, SampleCoverageError, SampleEntry
from eval.contracts.sample_results import validate_sample_metrics
from eval.sample_logging import canonicalize_sample
from eval.task import BaseBenchmark


class SampleEmissionError(RuntimeError):
    """A custom grader could not serialize a scored sample."""


class CustomSampleEmitter:
    """Bind generated examples to stable identities and append each completed score."""

    def __init__(
        self,
        task_name: str,
        benchmark: BaseBenchmark,
        generation_result: Mapping[str, Any],
        output_sink: FineStoreOutput,
    ) -> None:
        self.task_name = task_name
        self.benchmark = benchmark
        self.output_sink = output_sink
        self._examples: dict[int, tuple[str, int, int]] = {}
        self._seen: set[int] = set()
        self._entries: dict[tuple[str, int, int | None], SampleEntry] = {}
        self._sample_entries: dict[tuple[str, int], SampleEntry] = {}
        self._unit_repeats: dict[tuple[str, int], set[int | None]] = {}
        self._repeat_shape: tuple[int | None, ...] | None = None

        groups = benchmark.sample_groups(generation_result)
        position = 0
        for name, examples in groups.items():
            for ordinal, example in enumerate(examples):
                self._examples[id(example)] = (name, ordinal, position)
                position += 1

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
            name, ordinal, position = self._examples[id(example)]
            if id(example) in self._seen:
                raise SampleCoverageError(f"{self.task_name}: sample {position} was scored more than once")
            source = {name: [example]}
            records = self.benchmark.to_samples(source, {})
            if not records:
                raise SampleCoverageError(f"{self.task_name}: scored sample {position} has no sample record")
            repeat_shape = tuple(record.get("sample_repeat") for record in records)
            namespace = DEFAULT_SAMPLE_NAMESPACE if name == DEFAULT_SAMPLE_GROUP else name
            self._validate_repeats(namespace, ordinal, position, repeat_shape)
            for sample in records:
                sample["doc_id"] = position
                entry = self._sample_entry(namespace, ordinal, position, sample.get("sample_repeat"))
                record = canonicalize_sample(self.task_name, sample, position, entry)
                validate_sample_metrics(self.task_name, [record])
                self.output_sink.write_sample(self.task_name, record)
            self._seen.add(id(example))
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
            self.output_sink.finish_task()
        except Exception as exc:
            raise SampleEmissionError(f"{self.task_name}: failed to flush scored samples: {exc}") from exc
