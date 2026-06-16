"""Stage 4 — auto-detect CLI wiring tests (CPU).

Drives the REAL construction site (``eval.resume.wiring.build_resume_wiring`` +
``attach_to_chat_benchmarks``) — the single seam ``cli_evaluate`` uses to feed all
three resume paths — with a fake ``args`` Namespace and a fake LM, then runs the
real ``BaseBenchmark.compute`` to observe end-to-end behavior. The full
``cli_evaluate`` with a live vLLM model is the Stage-6 GPU gate; here we prove the
wiring decisions themselves:

  GATE 1 — ``--resume-mode off`` ⇒ no factory built, nothing attached, no
           ``resume/`` dir, ``compute`` byte-identical to the manager-absent run.
  GATE 2 — ``auto`` (default) FIRST run, NO prior state ⇒ manager auto-constructs
           but is a pure no-op: ``compute`` output is byte-identical to the
           manager-absent baseline and the ONLY new artifact is the inert
           fingerprint/state dir.
  GATE 3 — ``auto`` SECOND run with identical inputs ⇒ resumes (skipped > 0) and
           the resumed output equals the uninterrupted run.
  GATE 4 — a MATERIAL change against existing state ⇒ refuses (ResumeRefused).

Also: the 3b/3c seam — ``build_resume_wiring`` sets ``args.resume_manager_factory``
to a ``factory(task_name) -> ResumeManager`` that the lm-eval-native + pass@k paths
read (one construction site feeds all three — invariant #5).
"""

import argparse
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from eval.resume import ResumeManager, ResumeRefused, canonical_unit_key  # noqa: E402
from eval.resume.wiring import attach_to_chat_benchmarks, build_resume_wiring  # noqa: E402
from eval.task import BaseBenchmark  # noqa: E402


class _FakeLM:
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.generated_idxs = []

    def generate_until(self, instances):
        outs = []
        for inst in instances:
            self.generated_idxs.append(int(inst.idx))
            outs.append(f"out-{inst.idx}")
        return outs


class _Bench(BaseBenchmark):
    def generate_responses(self, model):  # pragma: no cover - unused
        return {}

    def evaluate_responses(self, results):  # pragma: no cover - unused
        return {}


class _FakeTaskManager:
    """Mimics InstructTaskManager's ``benchmark_instances`` mapping."""

    def __init__(self, instances):
        self.benchmark_instances = instances


def _instances(n):
    return [Instance("generate_until", {"i": i}, ("prompt", {"max_new_tokens": 8}), i) for i in range(n)]


def _args(tmp_path, mode="auto", output=True, **overrides):
    ns = argparse.Namespace(
        resume_mode=mode,
        output_path=str(tmp_path) if output else None,
        model_args="pretrained=some/model,revision=abc",
        model_name=None,
        max_tokens="256",
        gen_kwargs="temperature=0.7,top_p=1.0",
        num_samples=1,
        apply_chat_template=False,
        num_fewshot=None,
        seed=[0, 1234, 1234, 1234],
        annotator_model="auto",
        limit=None,
        predict_only=False,
        fewshot_as_multiturn=False,
        system_instruction=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ----- GATE 1: off ⇒ no-op, byte-identical -------------------------------------------------

def test_off_builds_no_factory_and_attaches_nothing(tmp_path):
    args = _args(tmp_path, mode="off")
    lm = _FakeLM()
    bench = _Bench()
    tm = _FakeTaskManager({"_Bench": bench})

    factory = build_resume_wiring(args, lm)
    attach_to_chat_benchmarks(tm, ["_Bench"], factory)

    assert factory is None
    assert args.resume_manager_factory is None
    assert bench._resume_manager is None  # nothing attached

    out = bench.compute(lm, _instances(4))
    assert out == [f"out-{i}" for i in range(4)]
    assert not (tmp_path / ".resume").exists()  # no state dir written


def test_no_output_path_disables_resume(tmp_path):
    args = _args(tmp_path, mode="auto", output=False)
    factory = build_resume_wiring(args, _FakeLM())
    assert factory is None
    assert args.resume_manager_factory is None


# ----- GATE 2: auto first run, no prior state ⇒ pure no-op (byte-identical) -----------------

def test_auto_first_run_no_prior_state_is_byte_identical(tmp_path):
    # Manager-ABSENT baseline.
    base_bench = _Bench()
    base_out = base_bench.compute(_FakeLM(), _instances(5))

    # Manager auto-constructed via the real wiring, FIRST run (no prior state).
    args = _args(tmp_path, mode="auto")
    lm = _FakeLM()
    bench = _Bench()
    tm = _FakeTaskManager({"_Bench": bench})
    factory = build_resume_wiring(args, lm)
    attach_to_chat_benchmarks(tm, ["_Bench"], factory)

    assert factory is not None
    assert args.resume_manager_factory is factory
    assert bench._resume_manager is not None  # constructed + attached

    out = bench.compute(lm, _instances(5))

    # THE Stage-4 invariant: output byte-identical to the manager-absent baseline.
    assert out == base_out
    # ... and every problem was actually generated (no false skips on a fresh run).
    assert lm.generated_idxs == [0, 1, 2, 3, 4]
    # The ONLY new artifact is the inert fingerprint/state dir.
    state_dir = tmp_path / ".resume"
    assert state_dir.exists()
    fp = list(state_dir.rglob("fingerprint.json"))
    assert len(fp) == 1  # exactly one per-task fingerprint written


# ----- GATE 3: auto second run ⇒ resumes (skipped > 0) -------------------------------------

def test_auto_second_run_resumes(tmp_path, caplog):
    import logging

    # Run 1: fresh, records all 5 units.
    args1 = _args(tmp_path, mode="auto")
    bench1 = _Bench()
    tm1 = _FakeTaskManager({"_Bench": bench1})
    f1 = build_resume_wiring(args1, _FakeLM())
    attach_to_chat_benchmarks(tm1, ["_Bench"], f1)
    lm1 = _FakeLM()
    out1 = bench1.compute(lm1, _instances(5))
    assert lm1.generated_idxs == [0, 1, 2, 3, 4]
    assert len(bench1._resume_manager.done_units()) == 5

    # Run 2: identical inputs -> auto-detects the prior state and resumes.
    args2 = _args(tmp_path, mode="auto")
    bench2 = _Bench()
    tm2 = _FakeTaskManager({"_Bench": bench2})
    f2 = build_resume_wiring(args2, _FakeLM())
    attach_to_chat_benchmarks(tm2, ["_Bench"], f2)
    assert bench2._resume_manager.decide() == "resume"

    lm2 = _FakeLM()
    with caplog.at_level(logging.INFO):
        out2 = bench2.compute(lm2, _instances(5))

    assert out2 == out1  # resumed output equals the uninterrupted run
    assert lm2.generated_idxs == []  # ALL skipped -> nothing regenerated (skipped > 0)
    assert any("skipped 5 done units" in r.message for r in caplog.records)


def test_auto_partial_resume_skips_done_only(tmp_path):
    # Run 1 records 3 of 5 (simulating a kill after 3).
    args1 = _args(tmp_path, mode="auto")
    bench1 = _Bench()
    f1 = build_resume_wiring(args1, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench1}), ["_Bench"], f1)
    bench1.compute(_FakeLM(), _instances(3))
    assert len(bench1._resume_manager.done_units()) == 3

    # Run 2: full 5 -> skips 0,1,2, regenerates only 3,4.
    args2 = _args(tmp_path, mode="auto")
    bench2 = _Bench()
    f2 = build_resume_wiring(args2, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench2}), ["_Bench"], f2)
    lm2 = _FakeLM()
    out2 = bench2.compute(lm2, _instances(5))
    assert out2 == [f"out-{i}" for i in range(5)]
    assert lm2.generated_idxs == [3, 4]


# ----- GATE 4: material delta ⇒ refuses ----------------------------------------------------

def test_auto_material_delta_refuses(tmp_path):
    # Run 1: establish state at seed [0,...].
    args1 = _args(tmp_path, mode="auto")
    bench1 = _Bench()
    f1 = build_resume_wiring(args1, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench1}), ["_Bench"], f1)
    bench1.compute(_FakeLM(), _instances(3))

    # Run 2: change a MATERIAL field (seed) against the same run dir -> refuse.
    args2 = _args(tmp_path, mode="auto", seed=[7, 7, 7, 7])
    bench2 = _Bench()
    f2 = build_resume_wiring(args2, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench2}), ["_Bench"], f2)
    with pytest.raises(ResumeRefused):
        bench2._resume_manager.decide()


def test_force_fresh_wipes_prior_state(tmp_path):
    # Run 1: record 3 units.
    args1 = _args(tmp_path, mode="auto")
    bench1 = _Bench()
    f1 = build_resume_wiring(args1, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench1}), ["_Bench"], f1)
    bench1.compute(_FakeLM(), _instances(3))
    assert len(bench1._resume_manager.done_units()) == 3

    # Run 2: force-fresh wipes the prior state and starts over.
    args2 = _args(tmp_path, mode="force-fresh")
    bench2 = _Bench()
    f2 = build_resume_wiring(args2, _FakeLM())
    attach_to_chat_benchmarks(_FakeTaskManager({"_Bench": bench2}), ["_Bench"], f2)
    assert bench2._resume_manager.decide() == "fresh"
    lm2 = _FakeLM()
    out2 = bench2.compute(lm2, _instances(3))
    assert lm2.generated_idxs == [0, 1, 2]  # all regenerated (state wiped)
    assert out2 == [f"out-{i}" for i in range(3)]


# ----- the 3b/3c seam: factory is the shared interface -------------------------------------

def test_factory_sets_resume_manager_factory_for_lm_eval_native_path(tmp_path):
    args = _args(tmp_path, mode="auto")
    factory = build_resume_wiring(args, _FakeLM())
    # The lm-eval-native (3b) + pass@k (3c) call sites read this attribute.
    assert args.resume_manager_factory is factory
    mgr = factory("gsm8k")
    assert isinstance(mgr, ResumeManager)
    assert mgr.mode == "auto"
    # per-task run dir isolation: two tasks get distinct run dirs.
    mgr2 = factory("MATH500")
    assert mgr.run_dir != mgr2.run_dir


def test_fingerprint_is_stable_across_identical_runs(tmp_path):
    a1 = _args(tmp_path, mode="auto")
    a2 = _args(tmp_path, mode="auto")
    fp1 = build_resume_wiring(a1, _FakeLM())("gsm8k")
    fp2 = build_resume_wiring(a2, _FakeLM())("gsm8k")
    assert fp1.fingerprint.value() == fp2.fingerprint.value()


def test_cosmetic_delta_does_not_change_fingerprint(tmp_path):
    a1 = _args(tmp_path, mode="auto", output=True)
    # change a cosmetic-only field (output_path subdir does not enter the hash; batch knobs absent).
    a2 = _args(tmp_path / "elsewhere", mode="auto")
    (tmp_path / "elsewhere").mkdir()
    fp1 = build_resume_wiring(a1, _FakeLM())("gsm8k")
    fp2 = build_resume_wiring(a2, _FakeLM())("gsm8k")
    assert fp1.fingerprint.value() == fp2.fingerprint.value()
