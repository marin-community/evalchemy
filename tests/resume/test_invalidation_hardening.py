"""Stage 5 — not-naive invalidation hardening (CPU / simulated multi-rank).

Hardens the edge cases that turn a "resume" into silent corruption (the difference
between a naive cache and Harbor's refuse-don't-corrupt discipline). Each edge is
CPU-testable; multi-rank is simulated with ``world_size>1`` + the per-rank manifest
layout (no GPU needed — real multi-GPU parity is Stage 6).

Edges (from ``stage5_invalidation_hardening_scope.md``):
  1. TP / rank-layout change vs per-rank state -> REFUSE with a clear error (decision #3),
     not a silent partial-merge. Tested BOTH directions: 1 -> 4 and 4 -> 1.
  3. Partial-unit corruption: trailing truncation -> truncate+regenerate (Stage 1); mid-file
     corruption -> RAISE; a corrupt ``fingerprint.json`` -> REFUSE (treat as unverifiable).
  4. Concurrent writers: N ranks appending to their per-rank files simultaneously -> no
     interleave / no corruption (realistic multi-rank case), and the resumed done-set is
     consistent.
  5. State-file placement: the ``.resume/`` state dir resolves under the run dir
     (``--output_path``), NEVER a package-relative or ``$HOME`` default — and works when
     ``$HOME`` is read-only (Leonardo ``HF_HUB_OFFLINE=1`` + RO-$HOME gotcha).

This module is pure stdlib (manager/manifest/fingerprint) so it runs on any Python (otagent).
"""

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from eval.resume import (
    ManifestWriter,
    ResumeManager,
    ResumeRefused,
    RunFingerprint,
    canonical_unit_key,
    read_manifest,
)
from eval.resume.manager import _fingerprint_path, _manifest_path, _resume_dir
from eval.resume.wiring import build_resume_wiring


def _fp():
    return RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 0.7})


# ============================================================================================
# Edge 1 — TP / rank-layout change vs per-rank state -> REFUSE (decision #3), both directions
# ============================================================================================

def test_rank_layout_1_to_4_refuses(tmp_path: Path):
    """A single-rank run, resumed under world_size=4 -> refuse (can't fan a single manifest out)."""
    run_dir = tmp_path / "run"
    fp = _fp()
    m1 = ResumeManager(run_dir, fp, mode="auto", world_size=1, rank=0)
    m1.decide()
    m1.record({"task": "T", "problem_idx": 0}, {"v": 0})
    assert (_resume_dir(run_dir) / "manifest.jsonl").exists()

    m4 = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=0)
    with pytest.raises(ResumeRefused) as exc:
        m4.decide()
    msg = str(exc.value).lower()
    assert "rank" in msg and "world_size=4" in str(exc.value)


def test_rank_layout_4_to_1_refuses(tmp_path: Path):
    """4 per-rank manifests, resumed under world_size=1 -> refuse.

    This is the DANGEROUS case: without the guard the world_size=1 run looks for
    ``manifest.jsonl`` (absent), finds an empty done set, and silently RE-RUNS everything
    (or worse, merges a disjoint slice). The guard must catch the prior per-rank layout.
    """
    run_dir = tmp_path / "run"
    fp = _fp()
    # Seed a 4-way run: each rank writes its own per-rank manifest.
    for r in range(4):
        m = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=r)
        m.decide()
        m.record({"task": "T", "rank": r, "problem_idx": r}, {"v": r})
    per_rank = sorted((_resume_dir(run_dir)).glob("manifest.jsonl.rank*"))
    assert len(per_rank) == 4
    assert not (_resume_dir(run_dir) / "manifest.jsonl").exists()

    # Now resume single-rank -> MUST refuse, not silently re-run.
    m1 = ResumeManager(run_dir, fp, mode="auto", world_size=1, rank=0)
    with pytest.raises(ResumeRefused) as exc:
        m1.decide()
    msg = str(exc.value)
    assert "rank" in msg.lower()
    assert "world_size=4" in msg and "world_size=1" in msg


def test_rank_layout_4_to_2_refuses(tmp_path: Path):
    """A different (non-trivial) rank count -> refuse (4-way slices don't align with 2-way)."""
    run_dir = tmp_path / "run"
    fp = _fp()
    for r in range(4):
        m = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=r)
        m.decide()
        m.record({"task": "T", "rank": r, "problem_idx": r}, {"v": r})

    m2 = ResumeManager(run_dir, fp, mode="auto", world_size=2, rank=0)
    with pytest.raises(ResumeRefused) as exc:
        m2.decide()
    s = str(exc.value)
    assert "world_size=4" in s and "world_size=2" in s


def test_matching_rank_layout_resumes(tmp_path: Path):
    """Same world_size -> per-rank state resumes normally (the guard does not false-positive)."""
    run_dir = tmp_path / "run"
    fp = _fp()
    for r in range(4):
        m = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=r)
        m.decide()
        m.record({"task": "T", "rank": r, "problem_idx": r}, {"v": r})

    # Resume rank 2 under the SAME world_size=4 -> resume, sees only its own unit.
    m_r2 = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=2)
    assert m_r2.decide() == "resume"
    assert m_r2.done_units() == {canonical_unit_key({"task": "T", "rank": 2, "problem_idx": 2})}


def test_force_fresh_overrides_rank_layout_refuse(tmp_path: Path):
    """force-fresh is the documented escape hatch from a rank-layout refuse."""
    run_dir = tmp_path / "run"
    fp = _fp()
    for r in range(4):
        m = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=r)
        m.decide()
        m.record({"task": "T", "rank": r, "problem_idx": r}, {"v": r})

    # force-fresh at world_size=1 wipes the prior per-rank state and starts clean.
    m1 = ResumeManager(run_dir, fp, mode="force-fresh", world_size=1, rank=0)
    assert m1.decide() == "fresh"
    assert m1.done_units() == set()
    assert not sorted((_resume_dir(run_dir)).glob("manifest.jsonl.rank*"))


# ============================================================================================
# Edge 3 — partial-unit corruption: trailing truncate vs mid-file vs corrupt fingerprint.json
# ============================================================================================

def test_trailing_truncation_truncates_and_regenerates(tmp_path: Path):
    """A truncated trailing line is skipped; that unit is NOT counted done (regenerated)."""
    run_dir = tmp_path / "run"
    fp = _fp()
    m = ResumeManager(run_dir, fp, mode="auto")
    m.decide()
    for i in range(3):
        m.record({"task": "T", "problem_idx": i}, {"v": i})
    # Simulate a SIGTERM mid-append on unit 3 (partial trailing line, no newline).
    manifest = _resume_dir(run_dir) / "manifest.jsonl"
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write('{"unit": {"task": "T", "problem_id')

    m2 = ResumeManager(run_dir, fp, mode="auto")
    assert m2.decide() == "resume"
    done = m2.done_units()
    assert done == {canonical_unit_key({"task": "T", "problem_idx": i}) for i in range(3)}
    # the half-written unit 3 is NOT done -> it will be regenerated
    assert m2.should_skip({"task": "T", "problem_idx": 3}) is False


def test_mid_file_corruption_raises(tmp_path: Path):
    """A corrupt line BEFORE a later valid line is real corruption -> RAISE (never lose middle units)."""
    run_dir = tmp_path / "run"
    fp = _fp()
    rdir = _resume_dir(run_dir)
    rdir.mkdir(parents=True)
    # a valid fingerprint.json forces decide() onto the resume path (so done_units reads the manifest)
    _fingerprint_path(run_dir).write_text(json.dumps({**fp.to_json(), "world_size": 1}))
    good = ManifestWriter(rdir / "manifest.jsonl")
    good.append({"task": "T", "problem_idx": 0}, {"v": 0})
    # inject a garbled COMPLETE (newline-terminated) line, then a valid one after it
    with open(rdir / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write("not-json-mid-file\n")
    good.append({"task": "T", "problem_idx": 2}, {"v": 2})

    with pytest.raises(ValueError):
        read_manifest(rdir / "manifest.jsonl")

    # ... and the manager surfaces it (does not silently swallow a mid-file corruption).
    m = ResumeManager(run_dir, fp, mode="auto")
    assert m.decide() == "resume"
    with pytest.raises(ValueError):
        m.done_units()


def test_corrupt_fingerprint_json_refuses(tmp_path: Path):
    """A garbled fingerprint.json cannot gate comparability -> REFUSE (not a silent resume/crash)."""
    run_dir = tmp_path / "run"
    rdir = _resume_dir(run_dir)
    rdir.mkdir(parents=True)
    # write a truncated / non-JSON fingerprint.json
    _fingerprint_path(run_dir).write_text('{"schema_version": 1, "value": "sha256:ab')

    m = ResumeManager(run_dir, _fp(), mode="auto")
    with pytest.raises(ResumeRefused) as exc:
        m.decide()
    s = str(exc.value).lower()
    assert "fingerprint" in s and "corrupt" in s


def test_fingerprint_json_missing_canonical_payload_refuses(tmp_path: Path):
    """A structurally-valid-JSON but schema-broken fingerprint.json (no canonical_payload) -> refuse."""
    run_dir = tmp_path / "run"
    rdir = _resume_dir(run_dir)
    rdir.mkdir(parents=True)
    _fingerprint_path(run_dir).write_text(json.dumps({"schema_version": 1, "value": "sha256:x"}))

    m = ResumeManager(run_dir, _fp(), mode="auto")
    with pytest.raises(ResumeRefused) as exc:
        m.decide()
    assert "fingerprint" in str(exc.value).lower()


def test_force_fresh_recovers_from_corrupt_fingerprint(tmp_path: Path):
    """force-fresh wipes a corrupt fingerprint.json and runs clean (the escape hatch)."""
    run_dir = tmp_path / "run"
    rdir = _resume_dir(run_dir)
    rdir.mkdir(parents=True)
    _fingerprint_path(run_dir).write_text("garbage")

    m = ResumeManager(run_dir, _fp(), mode="force-fresh")
    assert m.decide() == "fresh"
    # fingerprint.json rewritten + readable
    doc = json.loads(_fingerprint_path(run_dir).read_text())
    assert "canonical_payload" in doc


# ============================================================================================
# Edge 4 — concurrent writers (realistic multi-rank): no interleave + consistent resume
# ============================================================================================

def test_concurrent_multirank_append_no_corruption_and_resumes(tmp_path: Path):
    """W=4 ranks append to their per-rank files concurrently; each resumes a clean, complete set."""
    run_dir = tmp_path / "run"
    fp = _fp()
    world = 4
    n = 150
    mgrs = {r: ResumeManager(run_dir, fp, mode="auto", world_size=world, rank=r) for r in range(world)}
    for m in mgrs.values():
        m.decide()

    def writer(rank: int):
        m = mgrs[rank]
        for i in range(n):
            m.record({"task": "T", "rank": rank, "problem_idx": i}, {"v": i, "rank": rank})

    threads = [threading.Thread(target=writer, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each rank's manifest is complete + uncorrupted, and a fresh manager on the same dir
    # resumes the SAME clean set (no interleave between rank files).
    for r in range(world):
        states = read_manifest(_manifest_path(run_dir, world, r))  # raises if any line torn
        assert len(states) == n
        assert sorted(s.unit["problem_idx"] for s in states) == list(range(n))
        # all units belong to THIS rank (no cross-rank interleave into the file)
        assert all(s.unit["rank"] == r for s in states)

        resumed = ResumeManager(run_dir, fp, mode="auto", world_size=world, rank=r)
        assert resumed.decide() == "resume"
        assert len(resumed.done_units()) == n


def test_concurrent_appends_within_one_rank_file(tmp_path: Path):
    """Multiple threads on one ManifestWriter (n>1 sampling on a single rank) -> no torn lines."""
    path = tmp_path / "manifest.jsonl"
    w = ManifestWriter(path)
    n_threads, n_each = 8, 120

    def worker(tid: int):
        for i in range(n_each):
            w.append({"task": "T", "tid": tid, "sample_idx": i}, {"v": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    states = read_manifest(path)  # raises on any torn/corrupt line
    assert len(states) == n_threads * n_each


# ============================================================================================
# Edge 5 — state placement under output_path / RO-$HOME (Leonardo HF_HUB_OFFLINE gotcha)
# ============================================================================================

def test_state_dir_resolves_under_run_dir_not_home(tmp_path: Path):
    """The .resume state dir is anchored on the run_dir, never $HOME / a package-relative path."""
    run_dir = tmp_path / "work" / "myrun"
    fp = _fp()
    m = ResumeManager(run_dir, fp, mode="auto")
    m.decide()
    m.record({"task": "T", "problem_idx": 0}, {"v": 0})

    # state lives under the run dir
    assert _fingerprint_path(run_dir).exists()
    assert _manifest_path(run_dir, 1, 0).exists()
    # both paths are *under* run_dir (no escape to $HOME / cwd / package dir)
    assert Path(os.path.commonpath([run_dir, _fingerprint_path(run_dir)])) == run_dir
    assert Path(os.path.commonpath([run_dir, _manifest_path(run_dir, 1, 0)])) == run_dir
    # nothing was written to $HOME
    home = Path(os.path.expanduser("~"))
    assert not (home / "resume").exists()


def test_wiring_anchors_state_under_output_path(tmp_path):
    """build_resume_wiring places .resume strictly under --output_path (the writable work-FS run dir)."""
    pytest.importorskip("torch")
    pytest.importorskip("lm_eval")
    import argparse

    output = tmp_path / "work" / "out"
    output.mkdir(parents=True)
    args = argparse.Namespace(
        resume_mode="auto",
        output_path=str(output),
        model_args="pretrained=some/model,revision=abc",
        model_name=None,
        max_tokens="256",
        gen_kwargs="temperature=0.7,top_p=1.0",
        num_samples=1,
        apply_chat_template=False,
        num_fewshot=None,
        seed=[0, 1, 2, 3],
        annotator_model="auto",
        limit=None,
        predict_only=False,
        fewshot_as_multiturn=False,
        system_instruction=None,
    )

    class _LM:
        rank, world_size = 0, 1

    factory = build_resume_wiring(args, _LM())
    mgr = factory("gsm8k")
    # the manager's run dir is under <output_path>/.resume/...
    assert Path(os.path.commonpath([output, mgr.run_dir])) == output
    assert ".resume" in str(mgr.run_dir)


def test_state_writes_under_readonly_home(tmp_path, monkeypatch):
    """With $HOME set to a read-only dir (Leonardo), state still writes under the run dir.

    Simulates the Leonardo constraint: ``HF_HUB_OFFLINE=1`` + a read-only ``$HOME``. The
    resume state MUST live on the writable work-FS run dir (``--output_path``), so a RO $HOME
    must not break recording. We chmod a temp $HOME read-only, point HOME at it, and assert
    the run-dir state writes succeed and that NOTHING is written under the RO $HOME.
    """
    ro_home = tmp_path / "ro_home"
    ro_home.mkdir()
    # make $HOME read-only (no write/execute-create)
    ro_home.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("HOME", str(ro_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    work = tmp_path / "work" / "run"  # the writable work-FS run dir
    fp = _fp()
    try:
        m = ResumeManager(work, fp, mode="auto")
        assert m.decide() == "fresh"
        m.record({"task": "T", "problem_idx": 0}, {"v": 0})
        m.record({"task": "T", "problem_idx": 1}, {"v": 1})

        # state is on the work FS, fully functional under RO $HOME
        assert _fingerprint_path(work).exists()
        assert len(read_manifest(_manifest_path(work, 1, 0))) == 2

        # resume works too
        m2 = ResumeManager(work, fp, mode="auto")
        assert m2.decide() == "resume"
        assert len(m2.done_units()) == 2

        # NOTHING was written under the RO $HOME
        assert list(ro_home.iterdir()) == []
    finally:
        # restore writable perms so pytest can clean up tmp_path
        ro_home.chmod(stat.S_IRWXU)
