"""Durable per-request Evalchemy resume state in the evaluation's FineStore archive."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore
from rigging.filesystem.storage_path import prefix_join

from eval.contracts.resume_values import decode_resume_value, encode_resume_value
from eval.resume.fingerprint import RunFingerprint
from eval.resume.unit_keys import UnitKey, canonical_unit_key

RESUME_TABLE = "evalchemy_resume"
_FLUSH_EVERY = 32
_NAMESPACE = "namespace"
_UNIT_KEY = "unit_key"
_PAYLOAD = "payload"
_RESUME_SCHEMA = pa.schema(
    [pa.field(_NAMESPACE, pa.string()), pa.field(_UNIT_KEY, pa.string()), pa.field(_PAYLOAD, pa.string())]
)
ResumeMode = Literal["auto", "force-fresh"]


class ResumeRefused(RuntimeError):
    """Raised when committed FineStore state cannot safely be resumed."""


def _path_component(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value) or "task"


@dataclass
class FineStoreResumeManager:
    """Resume the existing unit-level evaluator protocol from committed FineStore rows."""

    root: str
    source_prefix: str
    task_name: str
    fingerprint: RunFingerprint
    mode: ResumeMode = "auto"
    world_size: int = 1
    rank: int = 0
    _store: DataStore | None = field(default=None, init=False, repr=False)
    _states: dict[UnitKey, dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _decision: str | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)

    @property
    def _namespace(self) -> str:
        return prefix_join(_path_component(self.source_prefix), _path_component(self.task_name))

    @property
    def _fingerprint_blob(self) -> str:
        return prefix_join(prefix_join("evalchemy", self._namespace), "resume/fingerprint.json")

    def decide(self) -> str:
        """Return fresh or resume after validating the stored fingerprint; refuse material changes."""
        if self._decision is not None:
            return self._decision
        view = ReadView(self.root)
        stored = view.read_blob(self._fingerprint_blob)
        if stored is None:
            store = self._open_store()
            document = {**self.fingerprint.to_json(), "world_size": self.world_size}
            with store.transaction() as transaction:
                transaction.write_object(self._fingerprint_blob, json.dumps(document, sort_keys=True).encode())
            self._decision = "fresh"
            self._states = {}
            return self._decision

        if self.mode == "force-fresh":
            raise ResumeRefused("FineStore output already exists; use a new output path for a fresh evaluation")
        try:
            document = json.loads(stored)
            prior = RunFingerprint(inputs=document["canonical_payload"])
            prior_world_size = int(document["world_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResumeRefused(f"invalid FineStore resume fingerprint for {self._namespace}") from exc
        if not self.fingerprint.matches(prior) or prior_world_size != self.world_size:
            raise ResumeRefused(f"FineStore resume fingerprint changed for {self._namespace}")
        self._decision = "resume"
        return self._decision

    def _open_store(self) -> DataStore:
        if self._store is None:
            self._store = DataStore.open(self.root, writer_id=f"evalchemy-resume-{uuid.uuid4().hex}")
            self._store.table(
                RESUME_TABLE,
                primary_key=(_NAMESPACE, _UNIT_KEY),
                schema=_RESUME_SCHEMA,
                on_conflict=OnConflict.ERROR,
            )
        return self._store

    def _load_states(self) -> dict[UnitKey, dict[str, Any]]:
        if self._states is not None:
            return self._states
        if self._decision is None:
            self.decide()
        namespace = prefix_join(self._namespace, f"rank{self.rank}")
        states = {}
        for row in ReadView(self.root).iter_rows(RESUME_TABLE, where=[(_NAMESPACE, "==", namespace)]):
            key = canonical_unit_key(json.loads(row[_UNIT_KEY]))
            states[key] = decode_resume_value(json.loads(row[_PAYLOAD]))
        self._states = states
        return states

    def done_units(self) -> set[UnitKey]:
        return set(self._load_states())

    def restore(self) -> dict[UnitKey, dict[str, Any]]:
        """Return committed payloads for this task and rank, keyed by evaluator request."""
        return dict(self._load_states())

    def should_skip(self, unit: dict[str, Any]) -> bool:
        return canonical_unit_key(unit) in self.done_units()

    def record(self, unit: dict[str, Any], payload: dict[str, Any]) -> None:
        if self._decision is None:
            self.decide()
        key = canonical_unit_key(unit)
        states = self._load_states()
        if key in states:
            return
        store = self._open_store()
        table = store.table(RESUME_TABLE, primary_key=(_NAMESPACE, _UNIT_KEY), schema=_RESUME_SCHEMA)
        table.append(
            {
                _NAMESPACE: prefix_join(self._namespace, f"rank{self.rank}"),
                _UNIT_KEY: json.dumps(unit, sort_keys=True),
                _PAYLOAD: json.dumps(encode_resume_value(payload), separators=(",", ":")),
            }
        )
        states[key] = payload
        self._pending_count += 1
        if self._pending_count >= _FLUSH_EVERY:
            store.flush_table(table)
            self._pending_count = 0

    def finalize(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
            self._pending_count = 0
