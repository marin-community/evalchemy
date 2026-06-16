"""Append-only JSONL manifest reader/writer for the resume manager.

The manifest is the per-unit *completion marker* (Stage-0 frozen artifact B). One line per
completed unit; append-with-flush so a SIGTERM at the wall loses at most the in-flight unit's
half-written trailing line, which is detected and skipped on read.

Design borrows from Harbor (anchors reconfirmed 2026-06-15 on `penfever/working`):
  - per-unit done marker, delete/skip partial   -> harbor `job.py:228-242`
  - atomic incremental write under a lock        -> harbor `job.py:443-449` + `:85`

Pure stdlib (no torch / lm_eval / numpy) so it is importable and unit-testable on any Python.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# A unit key is a small JSON-serializable dict, e.g. {"task": "MATH500", "problem_idx": 17}
# or {"task": "MATH500", "batch_idx": 3} for pass@k. We canonicalize it to a hashable tuple
# for set/dict membership while preserving the original dict in the manifest line.
UnitKey = Tuple[Tuple[str, Any], ...]


def canonical_unit_key(unit: Dict[str, Any]) -> UnitKey:
    """Map a unit dict to a deterministic hashable key (sorted by field name)."""
    return tuple(sorted((str(k), unit[k]) for k in unit))


def unit_key_to_dict(key: UnitKey) -> Dict[str, Any]:
    return {k: v for k, v in key}


@dataclass
class UnitState:
    """One completed unit as stored in the manifest."""

    unit: Dict[str, Any]
    payload: Dict[str, Any]
    ts: str

    @property
    def key(self) -> UnitKey:
        return canonical_unit_key(self.unit)

    def to_line(self) -> str:
        # sort_keys for stable, diff-friendly, content-deterministic lines.
        return json.dumps(
            {"unit": self.unit, "payload": self.payload, "ts": self.ts},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "UnitState":
        return cls(unit=obj["unit"], payload=obj["payload"], ts=obj.get("ts", ""))


class ManifestWriter:
    """Append-only writer for a single per-rank manifest file.

    `append()` writes one JSONL line and flushes + fsyncs so a crash cannot leave the line
    buffered. A process-level lock serializes concurrent `append()` calls within this process;
    per-rank file naming keeps separate ranks on separate files (Stage-0 decision #3 — a
    rank-layout change refuses rather than reusing another rank's file).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, unit: Dict[str, Any], payload: Dict[str, Any], ts: str | None = None) -> None:
        state = UnitState(
            unit=unit,
            payload=payload,
            ts=ts or datetime.now(timezone.utc).isoformat(),
        )
        line = state.to_line() + "\n"
        with self._lock:
            # Open in append+binary so we never truncate and the OS append is atomic per write
            # for line-sized payloads on a local FS. flush + fsync make the line durable.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())


def read_manifest(path: Path) -> List[UnitState]:
    """Read a manifest, dropping a corrupt / truncated *trailing* line.

    A SIGTERM mid-`append()` can leave a partial last line (no newline, or invalid JSON).
    We tolerate ONLY a corrupt *trailing* line: any corrupt line before a later valid line is a
    real corruption and raises (we never silently lose completed units in the middle). This
    mirrors Harbor treating a trial dir without `result.json` as not-done.
    """
    p = Path(path)
    if not p.exists():
        return []

    raw = p.read_text(encoding="utf-8", errors="strict")
    if raw == "":
        return []

    lines = raw.split("\n")
    # A well-formed file ends with "\n", so split yields a trailing "" — drop it. If the last
    # line is NOT empty, the file did not end in a newline => the last append was truncated.
    trailing_truncated = lines[-1] != ""
    if lines and lines[-1] == "":
        lines = lines[:-1]

    states: List[UnitState] = []
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        if line == "":
            # blank interior line: only tolerable as the very last record
            if is_last and trailing_truncated:
                break
            raise ValueError(f"{p}: unexpected blank line at index {idx}")
        try:
            obj = json.loads(line)
            states.append(UnitState.from_obj(obj))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if is_last and trailing_truncated:
                # The final line was being written when the process died: skip it.
                break
            raise ValueError(f"{p}: corrupt manifest line at index {idx}: {exc}") from exc
    return states


def iter_manifest(path: Path) -> Iterator[UnitState]:
    """Streaming view over a manifest (same corruption semantics as `read_manifest`)."""
    yield from read_manifest(path)
