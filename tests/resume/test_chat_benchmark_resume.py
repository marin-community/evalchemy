"""Stage 3a — chat_benchmark resume integration tests (CPU).

Exercises the ``BaseBenchmark.compute`` resume wrap (``_generate_with_resume`` +
``_unit_key`` + ``attach_resume_manager``) with a FAKE LM so no GPU/model is needed.
The two load-bearing properties:

  1. **Flag-off byte-identical:** with no manager attached (the default) OR a manager
     in ``off`` mode, ``compute`` returns exactly ``model.generate_until(prompts)`` and
     records nothing — byte-identical to today (global invariant #1).
  2. **Kill-and-resume parity:** a manager with prior manifest state skips the done
     problems (restores their stored output) and regenerates only the remainder; the
     merged result equals an uninterrupted run, in the same order.

``eval.task`` imports torch + lm_eval, so this whole module is skipped where those are
absent (it runs on the evalchemy / abb envs, not the pure-stdlib otagent env that the
rest of tests/resume/ runs on).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from eval.resume import (  # noqa: E402
    ManifestWriter,
    ResumeManager,
    RunFingerprint,
    canonical_unit_key,
)
from eval.task import BaseBenchmark  # noqa: E402


class _FakeLM:
    """Minimal stand-in for an lm_eval LM: deterministic generate_until + rank/world_size.

    ``generate_until`` returns a deterministic string per instance and records which
    instances it was asked to generate, so a test can assert that done problems were
    NOT regenerated.
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.generated_idxs = []

    def generate_until(self, instances):
        outs = []
        for inst in instances:
            self.generated_idxs.append((inst.idx, getattr(inst, "repeat_idx", None)))
            outs.append(f"out-{inst.idx}-{getattr(inst, 'repeat_idx', 0)}")
        return outs


class _Bench(BaseBenchmark):
    """Concrete BaseBenchmark whose abstract methods are stubbed (unused by these tests)."""

    def generate_responses(self, model):  # pragma: no cover - not exercised
        return {}

    def evaluate_responses(self, results):  # pragma: no cover - not exercised
        return {}


def _instances(n, repeat_idx=None):
    insts = []
    for i in range(n):
        inst = Instance("generate_until", {"i": i}, ("prompt", {"max_new_tokens": 8}), i)
        if repeat_idx is not None:
            inst.repeat_idx = repeat_idx
        insts.append(inst)
    return insts


def _fingerprint():
    return RunFingerprint.from_run_inputs(model_repo="m", task_name="_", task_data_digest="sha256:x")


# --- (1) flag-off byte-identical -------------------------------------------------------------

def test_no_manager_is_byte_identical():
    bench = _Bench()
    assert bench._resume_manager is None
    model = _FakeLM()
    out = bench.compute(model, _instances(5))
    assert out == [f"out-{i}-0" for i in range(5)]
    # generated every problem, recorded nothing (no manager)
    assert [g[0] for g in model.generated_idxs] == [0, 1, 2, 3, 4]


def test_off_mode_is_byte_identical_and_writes_nothing(tmp_path):
    bench = _Bench()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="off")
    bench.attach_resume_manager(mgr)
    model = _FakeLM()
    out = bench.compute(model, _instances(4))
    assert out == [f"out-{i}-0" for i in range(4)]
    assert [g[0] for g in model.generated_idxs] == [0, 1, 2, 3]
    # off mode is a pure no-op: no resume/ dir written
    assert not (tmp_path / "resume").exists()


def test_fresh_run_generates_all_and_records(tmp_path):
    bench = _Bench()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="auto")
    bench.attach_resume_manager(mgr)
    model = _FakeLM()
    out = bench.compute(model, _instances(3))
    assert out == ["out-0-0", "out-1-0", "out-2-0"]
    # all 3 generated (no prior state)
    assert [g[0] for g in model.generated_idxs] == [0, 1, 2]
    # all 3 recorded to the manifest
    assert len(mgr.done_units()) == 3


# --- (2) kill-and-resume parity --------------------------------------------------------------

def _seed_manifest(run_dir, task, done_idxs):
    """Simulate a killed run: pre-write completed units to the manifest."""
    mpath = run_dir / "resume" / "manifest.jsonl"
    w = ManifestWriter(mpath)
    for i in done_idxs:
        w.append({"task": task, "problem_idx": i}, {"output": f"out-{i}-0"})


def test_resume_skips_done_and_regenerates_remainder(tmp_path, caplog):
    task = "_"  # _Bench -> "_Bench".replace("Benchmark","") == "_Bench"; computed below
    # First, produce the canonical task_name the wrap uses:
    task = _Bench.__name__.replace("Benchmark", "")

    # Write the fingerprint that a prior run would have written (so auto resumes, not refuses).
    fp = _fingerprint()
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    import json

    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    _seed_manifest(tmp_path, task, done_idxs=[0, 1, 2])

    bench = _Bench()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    assert mgr.decide() == "resume"

    model = _FakeLM()
    import logging

    with caplog.at_level(logging.INFO):
        out = bench.compute(model, _instances(5))

    # done units restored, remainder regenerated, merged in order
    assert out == ["out-0-0", "out-1-0", "out-2-0", "out-3-0", "out-4-0"]
    # ONLY the remaining problems were actually generated
    assert [g[0] for g in model.generated_idxs] == [3, 4]
    # the skip log line fired with N>0
    assert any("skipped 3 done units" in r.message for r in caplog.records)


def test_resume_parity_equals_uninterrupted(tmp_path):
    task = _Bench.__name__.replace("Benchmark", "")

    # Baseline: uninterrupted fresh run.
    bench_a = _Bench()
    mgr_a = ResumeManager(run_dir=tmp_path / "a", fingerprint=_fingerprint(), mode="auto")
    bench_a.attach_resume_manager(mgr_a)
    baseline = bench_a.compute(_FakeLM(), _instances(6))

    # Killed-then-resumed: pre-seed half the units, then resume.
    import json

    rdir = tmp_path / "b" / "resume"
    rdir.mkdir(parents=True)
    fp = _fingerprint()
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    _seed_manifest(tmp_path / "b", task, done_idxs=[0, 1, 2])

    bench_b = _Bench()
    mgr_b = ResumeManager(run_dir=tmp_path / "b", fingerprint=fp, mode="auto")
    bench_b.attach_resume_manager(mgr_b)
    resumed = bench_b.compute(_FakeLM(), _instances(6))

    # Byte-identical (the fake generator is deterministic; greedy is deterministic for real).
    assert resumed == baseline


def test_aime24_repeat_idx_is_part_of_unit(tmp_path):
    """Each (problem, repeat) is a distinct unit so per-seed resume works (AIME24 n_repeat)."""
    task = _Bench.__name__.replace("Benchmark", "")
    fp = _fingerprint()
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    import json

    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    # Pre-complete repeat 0 entirely; repeat 1 not started.
    w = ManifestWriter(tmp_path / "resume" / "manifest.jsonl")
    for i in range(3):
        w.append({"task": task, "problem_idx": i, "repeat_idx": 0}, {"output": f"out-{i}-0"})

    bench = _Bench()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    assert mgr.decide() == "resume"
    model = _FakeLM()

    # repeat 0 -> all skipped
    out0 = bench.compute(model, _instances(3, repeat_idx=0))
    assert out0 == ["out-0-0", "out-1-0", "out-2-0"]
    assert model.generated_idxs == []  # nothing regenerated for repeat 0

    # repeat 1 -> all generated (distinct units)
    out1 = bench.compute(model, _instances(3, repeat_idx=1))
    assert out1 == ["out-0-1", "out-1-1", "out-2-1"]
    assert [g for g in model.generated_idxs] == [(0, 1), (1, 1), (2, 1)]


def test_unit_key_shape():
    bench = _Bench()
    inst = Instance("generate_until", {}, ("p", {}), 7)
    assert bench._unit_key("MATH500", inst) == {"task": "MATH500", "problem_idx": 7}
    inst.repeat_idx = 3
    assert bench._unit_key("AIME24", inst) == {"task": "AIME24", "problem_idx": 7, "repeat_idx": 3}
    # canonical key is hashable + sorted
    assert canonical_unit_key(bench._unit_key("AIME24", inst)) == (
        ("problem_idx", 7),
        ("repeat_idx", 3),
        ("task", "AIME24"),
    )
