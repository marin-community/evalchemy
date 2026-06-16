"""Stage-1 CPU unit tests for the standalone `eval/resume/` package.

Gate (from `notes/evalchemy/stage1_core_manager_scope.md`):
  (a) atomic append + read-back round-trip
  (b) corruption-skip: a truncated/garbled trailing line is dropped, prior units intact
  (c) force-fresh ignores existing state; `off` no-ops
  (d) fingerprint partition placeholder: cosmetic (IGNORED) fields excluded, material included
  (e) two concurrent per-rank writers don't interleave/corrupt
plus the auto-detect resume / material-delta refuse / rank-layout refuse decision rows.

The package is pure stdlib, so this runs on any Python with pytest (no GPU env needed).
"""

import json
import threading
from pathlib import Path

import pytest

from eval.resume import (
    IGNORED_FIELDS,
    MATERIAL_FIELDS,
    ManifestWriter,
    ResumeManager,
    ResumeRefused,
    RunFingerprint,
    canonical_unit_key,
    read_manifest,
)
from eval.resume.manifest import UnitState


# --------------------------------------------------------------------------------------------
# (a) atomic append + read-back round-trip
# --------------------------------------------------------------------------------------------
def test_append_readback_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    w = ManifestWriter(path)
    units = [
        ({"task": "MATH500", "problem_idx": i}, {"model_output": f"out{i}", "answer": str(i)})
        for i in range(5)
    ]
    for unit, payload in units:
        w.append(unit, payload)

    states = read_manifest(path)
    assert len(states) == 5
    for (unit, payload), state in zip(units, states):
        assert state.unit == unit
        assert state.payload == payload
        assert state.ts  # a timestamp was stamped
        assert state.key == canonical_unit_key(unit)


def test_unit_state_line_is_deterministic_and_sorted():
    s = UnitState(unit={"problem_idx": 3, "task": "MATH500"}, payload={"b": 2, "a": 1}, ts="T")
    line = s.to_line()
    # sort_keys => stable, diff-friendly content; round-trips through json.
    assert json.loads(line) == {"unit": {"problem_idx": 3, "task": "MATH500"},
                                "payload": {"a": 1, "b": 2}, "ts": "T"}
    assert line == UnitState.from_obj(json.loads(line)).to_line()


def test_read_missing_and_empty(tmp_path: Path):
    assert read_manifest(tmp_path / "nope.jsonl") == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert read_manifest(empty) == []


# --------------------------------------------------------------------------------------------
# (b) corruption-skip: truncated / garbled TRAILING line dropped; prior units intact
# --------------------------------------------------------------------------------------------
def test_corruption_skips_truncated_trailing_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    w = ManifestWriter(path)
    for i in range(3):
        w.append({"task": "T", "problem_idx": i}, {"v": i})
    # Simulate a SIGTERM mid-append: a partial trailing line with no newline.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"unit": {"task": "T", "problem_id')  # garbled, unterminated

    states = read_manifest(path)
    assert len(states) == 3  # the 3 complete units survive
    assert [s.unit["problem_idx"] for s in states] == [0, 1, 2]


def test_corruption_skips_garbled_trailing_line_with_newline(tmp_path: Path):
    # A trailing line that IS terminated by a newline but is invalid JSON is NOT auto-skipped:
    # only a truncated (no-trailing-newline) final write is tolerated; a complete-but-garbled
    # line is a real corruption mid-file and must raise (never silently lose middle units).
    path = tmp_path / "manifest.jsonl"
    w = ManifestWriter(path)
    w.append({"task": "T", "problem_idx": 0}, {"v": 0})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_interior_corruption_raises(tmp_path: Path):
    # A corrupt line BEFORE a later valid line is real corruption -> raise, don't silently drop.
    path = tmp_path / "manifest.jsonl"
    path.write_text("garbage-not-json\n" + UnitState({"task": "T", "problem_idx": 1}, {"v": 1}, "T").to_line() + "\n")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_resume_after_corruption_truncate_via_manager(tmp_path: Path):
    # End-to-end: a resumed run sees the prior complete units as done, ignoring the partial line.
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    mgr = ResumeManager(run_dir, fp, mode="auto")
    assert mgr.decide() == "fresh"
    for i in range(4):
        mgr.record({"task": "T", "problem_idx": i}, {"v": i})
    # corrupt the trailing line
    manifest = run_dir / "resume" / "manifest.jsonl"
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write('{"unit": {"task')

    mgr2 = ResumeManager(run_dir, fp, mode="auto")
    assert mgr2.decide() == "resume"
    done = mgr2.done_units()
    assert len(done) == 4
    assert all(mgr2.should_skip({"task": "T", "problem_idx": i}) for i in range(4))
    assert mgr2.should_skip({"task": "T", "problem_idx": 99}) is False


# --------------------------------------------------------------------------------------------
# (c) force-fresh ignores existing state; off no-ops
# --------------------------------------------------------------------------------------------
def test_force_fresh_wipes_existing_state(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    mgr = ResumeManager(run_dir, fp, mode="auto")
    mgr.decide()
    mgr.record({"task": "T", "problem_idx": 0}, {"v": 0})
    assert (run_dir / "resume" / "manifest.jsonl").exists()

    mgr2 = ResumeManager(run_dir, fp, mode="force-fresh")
    assert mgr2.decide() == "fresh"
    # state wiped: no done units, manifest gone (will be recreated on first record)
    assert mgr2.done_units() == set()
    assert not (run_dir / "resume" / "manifest.jsonl").exists()
    # fingerprint.json is rewritten fresh
    assert (run_dir / "resume" / "fingerprint.json").exists()


def test_off_is_pure_noop(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    mgr = ResumeManager(run_dir, fp, mode="off")
    assert mgr.decide() == "fresh"
    # off never reads/writes/skips
    assert mgr.should_skip({"task": "T", "problem_idx": 0}) is False
    mgr.record({"task": "T", "problem_idx": 0}, {"v": 0})
    assert mgr.done_units() == set()
    # NOTHING written to disk: pure no-op (byte-identical-to-today invariant)
    assert not (run_dir / "resume").exists()


def test_off_ignores_prior_state(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    # seed prior state
    seed = ResumeManager(run_dir, fp, mode="auto")
    seed.decide()
    seed.record({"task": "T", "problem_idx": 0}, {"v": 0})
    # off ignores it entirely
    mgr = ResumeManager(run_dir, fp, mode="off")
    mgr.decide()
    assert mgr.should_skip({"task": "T", "problem_idx": 0}) is False


# --------------------------------------------------------------------------------------------
# (d) fingerprint partition placeholder: cosmetic excluded, material included
# --------------------------------------------------------------------------------------------
def test_partition_lists_are_wired_and_disjoint():
    assert set(MATERIAL_FIELDS) & set(IGNORED_FIELDS) == set()
    # spot-check the stage-0 anchors are present in the right partition
    for m in ("model_repo", "model_revision", "task_data_digest", "temperature", "n",
              "num_samples", "seed_set", "num_fewshot", "passk_batch_size", "grader_version"):
        assert m in MATERIAL_FIELDS, f"{m} should be MATERIAL"
    for c in ("output_path", "job_name", "batch_size", "gpu_memory_utilization",
              "tensor_parallel_size", "world_size", "slurm_job_id"):
        assert c in IGNORED_FIELDS, f"{c} should be IGNORED"


def test_canonical_payload_excludes_cosmetic_includes_material():
    inputs = {
        # material
        "model_repo": "org/model", "task_name": "MATH500", "temperature": 0.7, "n": 8,
        # cosmetic
        "output_path": "/scratch/run1", "job_name": "abc", "batch_size": 256,
        "tensor_parallel_size": 4, "world_size": 4, "slurm_job_id": "12345",
    }
    fp = RunFingerprint(inputs=inputs)
    payload = fp._canonical_payload()
    for m in ("model_repo", "task_name", "temperature", "n"):
        assert m in payload
    for c in ("output_path", "job_name", "batch_size", "tensor_parallel_size",
              "world_size", "slurm_job_id"):
        assert c not in payload


def test_cosmetic_delta_yields_same_fingerprint():
    base = {"model_repo": "m", "task_name": "T", "temperature": 0.7}
    fp1 = RunFingerprint(inputs={**base, "output_path": "/a", "batch_size": 64,
                                 "tensor_parallel_size": 2, "world_size": 2})
    fp2 = RunFingerprint(inputs={**base, "output_path": "/b", "batch_size": 512,
                                 "tensor_parallel_size": 4, "world_size": 4})
    assert fp1.value() == fp2.value()
    assert fp1.matches(fp2)
    assert fp1.diff_material(fp2) == {}


def test_material_delta_yields_different_fingerprint():
    fp1 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 0.7})
    fp2 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 1.0})
    assert fp1.value() != fp2.value()
    assert not fp1.matches(fp2)
    assert "temperature" in fp1.diff_material(fp2)


def test_fingerprint_value_is_sha256_and_stable():
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    v = fp.value()
    assert v.startswith("sha256:") and len(v) == len("sha256:") + 64
    # key order in the input dict does not change the value (canonical sort)
    fp2 = RunFingerprint(inputs={"task_name": "T", "model_repo": "m"})
    assert fp.value() == fp2.value()


# --------------------------------------------------------------------------------------------
# decision table: auto resume / material refuse / rank-layout refuse
# --------------------------------------------------------------------------------------------
def test_auto_fresh_then_resume(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 0.7})
    m1 = ResumeManager(run_dir, fp, mode="auto")
    assert m1.decide() == "fresh"
    m1.record({"task": "T", "problem_idx": 0}, {"v": 0})
    m1.record({"task": "T", "problem_idx": 1}, {"v": 1})

    m2 = ResumeManager(run_dir, fp, mode="auto")
    assert m2.decide() == "resume"
    assert m2.done_units() == {canonical_unit_key({"task": "T", "problem_idx": i}) for i in (0, 1)}
    restored = m2.restore()
    assert restored[canonical_unit_key({"task": "T", "problem_idx": 0})] == {"v": 0}


def test_auto_refuses_on_material_delta(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp1 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 0.7})
    ResumeManager(run_dir, fp1, mode="auto").decide()

    fp2 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "temperature": 1.0})
    with pytest.raises(ResumeRefused) as exc:
        ResumeManager(run_dir, fp2, mode="auto").decide()
    assert "temperature" in str(exc.value)


def test_auto_resumes_despite_cosmetic_delta(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp1 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "output_path": "/a", "batch_size": 64})
    m1 = ResumeManager(run_dir, fp1, mode="auto")
    m1.decide()
    m1.record({"task": "T", "problem_idx": 0}, {"v": 0})

    fp2 = RunFingerprint(inputs={"model_repo": "m", "task_name": "T", "output_path": "/b", "batch_size": 512})
    m2 = ResumeManager(run_dir, fp2, mode="auto")
    assert m2.decide() == "resume"  # cosmetic-only delta still resumes


def test_rank_layout_mismatch_refuses(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    # seed a single-rank run
    m1 = ResumeManager(run_dir, fp, mode="auto", world_size=1, rank=0)
    m1.decide()
    m1.record({"task": "T", "problem_idx": 0}, {"v": 0})
    assert (run_dir / "resume" / "manifest.jsonl").exists()

    # resume under a multi-rank layout -> refuse (decision #3: merge deferred)
    m2 = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=0)
    with pytest.raises(ResumeRefused) as exc:
        m2.decide()
    assert "rank" in str(exc.value).lower()


def test_per_rank_files_are_separate(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    m_r0 = ResumeManager(run_dir, fp, mode="auto", world_size=2, rank=0)
    m_r1 = ResumeManager(run_dir, fp, mode="auto", world_size=2, rank=1)
    m_r0.decide()
    m_r1.decide()
    m_r0.record({"task": "T", "problem_idx": 0}, {"v": 0})
    m_r1.record({"task": "T", "problem_idx": 1}, {"v": 1})
    assert (run_dir / "resume" / "manifest.jsonl.rank0").exists()
    assert (run_dir / "resume" / "manifest.jsonl.rank1").exists()
    # each rank only sees its own units
    assert m_r0.done_units() == {canonical_unit_key({"task": "T", "problem_idx": 0})}
    assert m_r1.done_units() == {canonical_unit_key({"task": "T", "problem_idx": 1})}


# --------------------------------------------------------------------------------------------
# (e) concurrent writers don't interleave / corrupt
# --------------------------------------------------------------------------------------------
def test_concurrent_writers_no_corruption(tmp_path: Path):
    # Two ranks writing their own per-rank files concurrently (the production layout under TP).
    run_dir = tmp_path / "run"
    fp = RunFingerprint(inputs={"model_repo": "m", "task_name": "T"})
    mgrs = {r: ResumeManager(run_dir, fp, mode="auto", world_size=2, rank=r) for r in (0, 1)}
    for m in mgrs.values():
        m.decide()

    n = 200

    def writer(rank: int):
        m = mgrs[rank]
        for i in range(n):
            m.record({"task": "T", "rank": rank, "problem_idx": i}, {"v": i})

    threads = [threading.Thread(target=writer, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in (0, 1):
        states = read_manifest(run_dir / "resume" / f"manifest.jsonl.rank{r}")
        assert len(states) == n
        assert sorted(s.unit["problem_idx"] for s in states) == list(range(n))


def test_concurrent_appends_single_file(tmp_path: Path):
    # The in-process lock serializes concurrent appends to one ManifestWriter (no torn lines).
    path = tmp_path / "manifest.jsonl"
    w = ManifestWriter(path)
    n_threads, n_each = 8, 100

    def worker(tid: int):
        for i in range(n_each):
            w.append({"task": "T", "tid": tid, "i": i}, {"v": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    states = read_manifest(path)  # raises if any line is torn/corrupt
    assert len(states) == n_threads * n_each
