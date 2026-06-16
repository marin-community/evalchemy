"""ResumeManager — the single unified interface all three eval paths call (Stage-0 frozen).

One interface, three paths (global invariant #5). The only per-path difference is the unit-key
schema (`{task, problem_idx}` for chat_benchmark/lm-eval, `{task, batch_idx}` for pass@k) and
where `record()` is called. Native pass@k gets NO special-casing (decision #4).

Two-tier design mirroring Harbor (anchors reconfirmed 2026-06-15):
  - strict refuse gate (material fingerprint mismatch -> refuse)      harbor `job.py:218-226`
  - tolerant canonical fingerprint                                    harbor `lock.py:127-149`
  - per-unit done marker (manifest line) + corrupt-trailing skip      harbor `job.py:228-242`
  - remaining = planned minus done                                    harbor `job.py:277-307`

Modes (`--resume-mode`, plumbed as a constructor arg this stage; CLI wiring is Stage 4):
  - "auto"       : detect prior state; resume on fingerprint match, REFUSE on material delta.
  - "force-fresh": wipe any prior state and run fresh (ignores existing state).
  - "off"        : ignore prior state, do NOT skip, do NOT write — pure no-op for byte-identical
                   baselining (global invariant #1).

This module is standalone in Stage 1 — NOTHING in production imports it yet, so the flag-off /
no-prior-state byte-identical invariant is automatically true. Pure stdlib.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set

from .fingerprint import RunFingerprint
from .manifest import (
    ManifestWriter,
    UnitKey,
    canonical_unit_key,
    read_manifest,
)

ResumeMode = Literal["auto", "force-fresh", "off"]
Decision = Literal["fresh", "resume", "refuse"]


class ResumeRefused(RuntimeError):
    """Raised when a resume is refused (material fingerprint delta, or rank-layout mismatch).

    The evalchemy analogue of Harbor's `FileExistsError` (`job.py:218-226`) — refuse loudly
    rather than silently resume into a comparability-breaking mix (global invariant #3).
    """


def _resume_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "resume"


def _fingerprint_path(run_dir: Path) -> Path:
    return _resume_dir(run_dir) / "fingerprint.json"


def _manifest_path(run_dir: Path, world_size: int, rank: int) -> Path:
    base = _resume_dir(run_dir) / "manifest.jsonl"
    if world_size > 1:
        # per-rank file, mirroring lm-eval `_rank<R>.db`; a rank-layout change refuses (decision #3).
        return base.with_name(f"manifest.jsonl.rank{rank}")
    return base


@dataclass
class ResumeManager:
    """See module docstring. Frozen signature from Stage-0 §"Unified interface"."""

    run_dir: Path
    fingerprint: RunFingerprint
    mode: ResumeMode = "auto"
    world_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self._decision: Optional[Decision] = None
        self._writer: Optional[ManifestWriter] = None
        self._done: Optional[Set[UnitKey]] = None
        if self.mode not in ("auto", "force-fresh", "off"):
            raise ValueError(f"unknown resume mode: {self.mode!r}")

    # ---- entry point ------------------------------------------------------------------------
    def decide(self) -> Decision:
        """Detect prior state, validate the fingerprint, and pick fresh/resume/refuse.

        Side effect: writes `fingerprint.json` on a fresh/force-fresh run, validates it on
        resume. Returns the decision; idempotent within an instance.
        """
        if self._decision is not None:
            return self._decision

        if self.mode == "off":
            # Pure no-op: never read, never write, never skip. Byte-identical to today.
            self._decision = "fresh"
            self._done = set()
            return self._decision

        fp_path = _fingerprint_path(self.run_dir)

        if self.mode == "force-fresh":
            # Wipe any prior state and start clean.
            rdir = _resume_dir(self.run_dir)
            if rdir.exists():
                shutil.rmtree(rdir)
            self._write_fingerprint()
            self._decision = "fresh"
            self._done = set()
            return self._decision

        # mode == "auto"
        if not fp_path.exists():
            # No prior state -> fresh (byte-identical to today).
            self._write_fingerprint()
            self._decision = "fresh"
            self._done = set()
            return self._decision

        # Prior state exists -> validate fingerprint. A corrupt/truncated fingerprint.json
        # cannot be trusted to gate comparability, so we REFUSE rather than resume into an
        # unverifiable prior state (fail-safe toward refuse — global invariant #3). force-fresh
        # is the documented escape hatch.
        try:
            prior_doc = json.loads(fp_path.read_text())
            prior = RunFingerprint(inputs=prior_doc["canonical_payload"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            raise ResumeRefused(
                f"Run dir {self.run_dir} has an existing but corrupt/unreadable "
                f"fingerprint.json ({fp_path}): {exc}. Cannot verify comparability; refusing "
                f"to resume. Use --resume-mode force-fresh to start over."
            ) from exc
        if not self.fingerprint.matches(prior):
            changed = self.fingerprint.diff_material(prior)
            raise ResumeRefused(
                f"Run dir {self.run_dir} has an existing run with a different material "
                f"fingerprint; refusing to resume. Changed fields: {changed}. "
                f"Use --resume-mode force-fresh to start over."
            )

        # rank-layout guard (decision #3): the run's world_size must match the prior run's
        # recorded world_size; a mismatch refuses rather than resuming against an unmergeable
        # per-rank layout (per-rank slices are disjoint `islice`s — merge is deferred).
        self._guard_rank_layout(prior_doc)
        self._decision = "resume"
        self._done = None  # lazily loaded by done_units()
        return self._decision

    def _guard_rank_layout(self, prior_doc: Dict[str, Any]) -> None:
        """Refuse any resume whose world_size differs from the prior run's world_size.

        The per-rank manifest layout encodes the rank count: a single-rank run writes
        ``manifest.jsonl``; a W-way run writes ``manifest.jsonl.rankR`` for R in [0, W). The
        AUTHORITATIVE prior rank count is the ``world_size`` stamped into ``fingerprint.json``
        at fresh time — inferring it from the on-disk per-rank files is unreliable because each
        rank writes its file lazily (a partially-complete W-way run has < W files). A resume
        whose ``world_size`` differs from the prior cannot be safely merged (decision #3, merge
        deferred), so we REFUSE loudly rather than silently resume against the wrong slice or an
        empty done set. Guards every direction: 1->W, W->1 (the dangerous silent-rerun case),
        and W->W'.
        """
        prior_world = prior_doc.get("world_size")
        if prior_world is None:
            # Backward-compat: a fingerprint.json written before world_size provenance existed.
            # Fall back to the on-disk layout CLASS (single vs per-rank), which is unambiguous
            # even if the exact prior W is unknown.
            rdir = _resume_dir(self.run_dir)
            prior_is_multi = bool(sorted(rdir.glob("manifest.jsonl.rank*")))
            prior_is_single = (rdir / "manifest.jsonl").exists()
            now_is_multi = self.world_size > 1
            if (prior_is_multi and not now_is_multi) or (prior_is_single and now_is_multi):
                raise ResumeRefused(
                    f"Run dir {self.run_dir} has a "
                    f"{'multi' if prior_is_multi else 'single'}-rank manifest layout but this "
                    f"run has world_size={self.world_size}; rank-layout mismatch. Refusing "
                    f"(merge deferred). Use --resume-mode force-fresh."
                )
            return

        if int(prior_world) != int(self.world_size):
            raise ResumeRefused(
                f"Run dir {self.run_dir} was created with world_size={prior_world} but this "
                f"run has world_size={self.world_size}; rank-layout mismatch — per-rank slices "
                f"don't align and merge is deferred (decision #3). Refusing rather than "
                f"silently resuming against the wrong layout. Use --resume-mode force-fresh."
            )

    def _write_fingerprint(self) -> None:
        fp_path = _fingerprint_path(self.run_dir)
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.fingerprint.to_json()
        # Stamp the rank count as provenance (NOT part of the content hash — world_size is an
        # IGNORED field for comparability) so a later resume can authoritatively detect a
        # rank-layout change (decision #3). Mirrors Harbor preserving provenance on lock rewrite.
        doc["world_size"] = int(self.world_size)
        fp_path.write_text(json.dumps(doc, indent=2, sort_keys=True))

    # ---- unified interface ------------------------------------------------------------------
    def done_units(self) -> Set[UnitKey]:
        """Set of completed unit keys from the manifest (corrupt trailing line skipped)."""
        if self._done is not None:
            return self._done
        if self._decision is None:
            self.decide()
        if self._decision != "resume":
            self._done = set()
            return self._done
        states = read_manifest(_manifest_path(self.run_dir, self.world_size, self.rank))
        self._done = {s.key for s in states}
        return self._done

    def restore(self) -> Dict[UnitKey, Dict[str, Any]]:
        """Map each completed unit key -> its payload (for reconstructing in-memory results)."""
        if self._decision is None:
            self.decide()
        if self._decision != "resume":
            return {}
        states = read_manifest(_manifest_path(self.run_dir, self.world_size, self.rank))
        return {s.key: s.payload for s in states}

    def should_skip(self, unit: Dict[str, Any]) -> bool:
        """True if this unit is already recorded as done (so the path skips regenerating it)."""
        if self.mode == "off":
            return False
        return canonical_unit_key(unit) in self.done_units()

    def record(self, unit: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Atomically append a completed unit to the manifest (no-op when mode == off)."""
        if self.mode == "off":
            return
        if self._decision is None:
            self.decide()
        if self._writer is None:
            self._writer = ManifestWriter(_manifest_path(self.run_dir, self.world_size, self.rank))
        self._writer.append(unit, payload)
        # keep the in-memory done set coherent so should_skip sees it immediately
        if self._done is not None:
            self._done.add(canonical_unit_key(unit))

    def finalize(self) -> None:
        """Optional close hook. Append+fsync per record means nothing is buffered; no-op today."""
        return None
