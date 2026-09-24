"""Shared grader scheduling and generation-artifact lifecycle contracts."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

GENERATION_ARTIFACT_SCHEMA_VERSION = 1

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


class GraderExecutionMode(StrEnum):
    """Isolation a grader requires from the Evalchemy driver."""

    SERIAL = "serial"
    THREAD_SAFE = "thread_safe"
    PROCESS_ISOLATED = "process_isolated"
    SANDBOXED = "sandboxed"


class ArtifactValidationError(RuntimeError):
    """A required generation artifact is absent, malformed, or incomplete."""


@dataclass(frozen=True)
class GenerationArtifact:
    """One immutable JSONL artifact registered with its expected contents."""

    name: str
    path: Path
    expected_count: int
    sha256: str

    def to_dict(self, root: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path.relative_to(root)),
            "expected_count": self.expected_count,
            "sha256": self.sha256,
        }


class GenerationArtifactManifest:
    """Own temporary grader inputs from atomic write through cleanup."""

    def __init__(self, root: Path, owner: tempfile.TemporaryDirectory[str] | None = None):
        self.root = root.resolve()
        self._owner = owner
        self._artifacts: dict[str, GenerationArtifact] = {}

    @classmethod
    def temporary(cls) -> "GenerationArtifactManifest":
        owner = tempfile.TemporaryDirectory(prefix="evalchemy-grading-")
        return cls(Path(owner.name), owner)

    def write_jsonl(
        self,
        name: str,
        relative_path: str,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_count: int,
    ) -> GenerationArtifact:
        """Atomically write and register a required JSONL artifact."""
        if not name or name in self._artifacts:
            raise ArtifactValidationError(f"artifact name must be unique and non-empty: {name!r}")
        if expected_count < 0 or len(records) != expected_count:
            raise ArtifactValidationError(
                f"artifact {name} expected {expected_count} records but received {len(records)}"
            )
        target = (self.root / relative_path).resolve()
        if target.parent != self.root or target.name != relative_path:
            raise ArtifactValidationError("artifact path must be a filename directly under its manifest root")

        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=self.root)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            os.replace(temporary_path, target)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        artifact = GenerationArtifact(name, target, expected_count, digest)
        self._artifacts[name] = artifact
        self.validate(name)
        return artifact

    def path(self, name: str) -> Path:
        """Return a validated artifact path for a grader invocation."""
        self.validate(name)
        return self._artifacts[name].path

    def validate(self, name: str) -> None:
        artifact = self._artifacts.get(name)
        if artifact is None:
            raise ArtifactValidationError(f"required artifact is not registered: {name}")
        if not artifact.path.is_file():
            raise ArtifactValidationError(f"required artifact is missing: {name}")
        try:
            # JSONL records are separated by LF; splitlines() also treats legal
            # Unicode line and paragraph separators inside JSON strings as boundaries.
            lines = artifact.path.read_text(encoding="utf-8").split("\n")
            if lines[-1] == "":
                lines.pop()
            records = [json.loads(line) for line in lines]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"required artifact is invalid JSONL: {name}") from exc
        if len(records) != artifact.expected_count:
            raise ArtifactValidationError(
                f"artifact {name} expected {artifact.expected_count} records but contains {len(records)}"
            )
        digest = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ArtifactValidationError(f"required artifact changed after registration: {name}")

    def validate_required(self) -> None:
        """Validate every artifact immediately before grading."""
        if not self._artifacts:
            raise ArtifactValidationError("generation artifact manifest has no required artifacts")
        for name in self._artifacts:
            self.validate(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_ARTIFACT_SCHEMA_VERSION,
            "artifacts": [artifact.to_dict(self.root) for artifact in self._artifacts.values()],
        }

    def cleanup(self) -> None:
        """Release all lifecycle-owned files; safe to call repeatedly."""
        if self._owner is not None:
            self._owner.cleanup()
            self._owner = None


def generation_artifacts(generation_result: Any) -> GenerationArtifactManifest | None:
    """Return the lifecycle-owned artifact manifest from a generation result."""
    if not isinstance(generation_result, Mapping):
        return None
    manifest = generation_result.get("artifacts")
    return manifest if isinstance(manifest, GenerationArtifactManifest) else None


def validate_serialized_artifact_manifests(serialized: Any) -> None:
    """Validate persisted artifact manifests without requiring cleaned-up files."""
    if not isinstance(serialized, Mapping):
        raise TypeError("generation_artifacts must be a mapping")
    for task_name, manifest in serialized.items():
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("generation artifact task names must be non-empty strings")
        if not isinstance(manifest, Mapping):
            raise TypeError("generation artifact manifest must be a mapping")
        if manifest.get("schema_version") != GENERATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"generation artifact schema_version must be {GENERATION_ARTIFACT_SCHEMA_VERSION}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or not artifacts:
            raise ValueError("generation artifact manifest must contain artifacts")
        names: set[str] = set()
        paths: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TypeError("generation artifact must be a mapping")
            name = artifact.get("name")
            path = artifact.get("path")
            count = artifact.get("expected_count")
            digest = artifact.get("sha256")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("generation artifact names must be unique and non-empty")
            if not isinstance(path, str) or not path or path in paths or Path(path).name != path:
                raise ValueError("generation artifact paths must be unique filenames")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("generation artifact expected_count must be a non-negative integer")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("generation artifact sha256 must be a lowercase hexadecimal digest")
            names.add(name)
            paths.add(path)


def execute_grading_jobs(
    jobs: Sequence[_Job],
    *,
    mode_for: Callable[[_Job], GraderExecutionMode],
    grade: Callable[[_Job], _Result],
    max_workers: int | None = None,
) -> list[_Result]:
    """Execute grader jobs under their declared isolation while preserving order."""
    results: dict[int, _Result] = {}
    thread_jobs: list[tuple[int, _Job]] = []
    process_jobs: list[tuple[int, _Job]] = []

    for index, job in enumerate(jobs):
        mode = GraderExecutionMode(mode_for(job))
        if mode is GraderExecutionMode.THREAD_SAFE:
            thread_jobs.append((index, job))
        elif mode is GraderExecutionMode.PROCESS_ISOLATED:
            process_jobs.append((index, job))
        else:
            # SERIAL graders and graders that own a sandbox may create processes or
            # mutate process globals, so they never enter the driver's thread pool.
            results[index] = grade(job)

    if thread_jobs:
        workers = max_workers or min(len(thread_jobs), (os.cpu_count() or 1) * 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(grade, job): index for index, job in thread_jobs}
            for future, index in ((future, futures[future]) for future in futures):
                results[index] = future.result()

    if process_jobs:
        workers = max_workers or min(len(process_jobs), os.cpu_count() or 1)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(grade, job): index for index, job in process_jobs}
            for future, index in ((future, futures[future]) for future in futures):
                results[index] = future.result()

    return [results[index] for index in range(len(jobs))]
