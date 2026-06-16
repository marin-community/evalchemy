"""Stage 3c — native pass@k resume-integration tests (CPU).

Exercises ``BaseBenchmark.generate_n_samples_batched`` — the resume wrap around the
Stage-2b native pass@k generation, at unit ``{task, batch_idx}`` (problem-batches of
size ``B``, full ``num_samples`` per problem, no n-split). Uses a FAKE LM so no
GPU/model is needed. The load-bearing properties:

  1. **Flag-off byte-identical to Stage 2b:** with no manager attached (the default),
     ``generate_n_samples_batched`` returns exactly what ``generate_n_samples`` returns
     and records nothing — byte-identical to Stage-2b output (global invariant #1).
  2. **Kill-and-resume parity (exact):** a manager with prior manifest state skips the
     done problem-batches (restores their per-(problem,sample) outputs) and regenerates
     only the remaining batches; the merged per-problem outputs AND the aggregated
     pass@k table equal an uninterrupted run, identical by construction (no n-split).
  3. **Unit-key + restore/merge:** units are ``{task, batch_idx}``; the payload carries
     per-(problem,sample) outputs keyed by problem index.

``eval.task`` imports torch + lm_eval, so this whole module is skipped where those are
absent (it runs on the evalchemy / abb envs, not the pure-stdlib otagent env).
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from eval.passk import aggregate_pass_at_k  # noqa: E402
from eval.resume import ManifestWriter, ResumeManager, RunFingerprint  # noqa: E402
from eval.task import BaseBenchmark  # noqa: E402


class _FakeLM:
    """Deterministic stand-in: ``generate_until`` returns a string per (problem, sample).

    The output for a given (problem_idx, repeat_idx) is the SAME no matter which batch
    it lands in or how the problems are sliced — this mirrors vLLM's per-request seeding
    (decision #4: full num_samples per problem in one engine call, no batch-composition
    dependence), which is exactly the property Stage-3c parity relies on. It also records
    which (problem, sample) pairs it was actually asked to generate.
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.generated = []  # list of (problem_idx, repeat_idx)

    def generate_until(self, instances):
        outs = []
        for inst in instances:
            r = getattr(inst, "repeat_idx", 0)
            self.generated.append((inst.idx, r))
            outs.append(f"sol-{inst.idx}-{r}")
        return outs


class _Bench(BaseBenchmark):
    """Concrete BaseBenchmark; abstract methods stubbed (unused by these tests)."""

    def generate_responses(self, model):  # pragma: no cover - not exercised
        return {}

    def evaluate_responses(self, results):  # pragma: no cover - not exercised
        return {}


def _build_instances_factory(num_problems):
    """Return a ``build_instances(sample_idx, seed)`` callback over ``num_problems``."""

    def build_instances(sample_idx, seed):
        insts = []
        for i in range(num_problems):
            inst = Instance("generate_until", {"i": i}, ("prompt", {"max_new_tokens": 8, "seed": seed}), i)
            inst.repeat_idx = sample_idx
            insts.append(inst)
        return insts

    return build_instances


def _fingerprint(B=None):
    return RunFingerprint.from_run_inputs(
        model_repo="m", task_name="_", task_data_digest="sha256:x", passk_batch_size=B
    )


TASK = _Bench.__name__.replace("Benchmark", "")  # "_"


# --- (1) flag-off byte-identical to Stage 2b -------------------------------------------------

def test_no_manager_equals_generate_n_samples():
    """Flag-off: batched path == the Stage-2b generate_n_samples output, byte-identical."""
    num_problems, n = 5, 4
    bif = _build_instances_factory(num_problems)

    bench_a = _Bench(num_samples=n)
    assert bench_a._resume_manager is None
    out_batched = bench_a.generate_n_samples_batched(_FakeLM(), bif, n)

    bench_b = _Bench(num_samples=n)
    out_plain = bench_b.generate_n_samples(_FakeLM(), bif, n)

    assert out_batched == out_plain
    # shape: one list of n completions per problem
    assert len(out_batched) == num_problems
    assert all(len(p) == n for p in out_batched)
    assert out_batched[2] == ["sol-2-0", "sol-2-1", "sol-2-2", "sol-2-3"]


def test_no_manager_writes_nothing(tmp_path):
    bench = _Bench(num_samples=3)
    bench.generate_n_samples_batched(_FakeLM(), _build_instances_factory(4), 3)
    assert not (tmp_path / "resume").exists()


# --- (2)/(3) kill-and-resume parity + unit-key/restore ---------------------------------------

def _uninterrupted_table(num_problems, n, B):
    """Fresh batched run -> per-problem outputs + the pass@k table (the baseline)."""
    bench = _Bench(num_samples=n)
    bench.passk_batch_size = B
    out = bench.generate_n_samples_batched(_FakeLM(), _build_instances_factory(num_problems), n)
    # grade: a sample is "correct" iff its repeat_idx is even (deterministic toy grader)
    num_correct = [sum(1 for s in per if int(s.rsplit("-", 1)[1]) % 2 == 0) for per in out]
    table = aggregate_pass_at_k(n, num_correct, [1, 2, 4])
    return out, table


def test_resume_skips_done_batches_and_regenerates_remainder(tmp_path, caplog):
    import logging

    num_problems, n, B = 6, 4, 2  # 3 batches of 2 problems
    fp = _fingerprint(B)
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))

    # Pre-complete batch 0 (problems 0,1) by writing the manifest as a killed run would.
    w = ManifestWriter(tmp_path / "resume" / "manifest.jsonl")
    batch0_outputs = {str(p): [f"sol-{p}-{r}" for r in range(n)] for p in (0, 1)}
    w.append({"task": TASK, "batch_idx": 0}, {"outputs": batch0_outputs})

    bench = _Bench(num_samples=n)
    bench.passk_batch_size = B
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    assert mgr.decide() == "resume"

    model = _FakeLM()
    with caplog.at_level(logging.INFO):
        out = bench.generate_n_samples_batched(model, _build_instances_factory(num_problems), n)

    # batch 0 restored, not regenerated; batches 1,2 (problems 2..5) regenerated
    assert sorted(set(p for p, _ in model.generated)) == [2, 3, 4, 5]
    assert all((0, r) not in model.generated and (1, r) not in model.generated for r in range(n))
    # final per-problem outputs correct for ALL problems (restored U new)
    assert out[0] == [f"sol-0-{r}" for r in range(n)]
    assert out[5] == [f"sol-5-{r}" for r in range(n)]
    # skip log fired with N>0
    assert any("skipped 1 done batches" in r.message for r in caplog.records)


def test_resume_table_equals_uninterrupted(tmp_path):
    """The load-bearing gate: resumed pass@k table == uninterrupted pass@k table (exact)."""
    num_problems, n, B = 6, 4, 2

    # Baseline: uninterrupted fresh batched run (no manager).
    baseline_out, baseline_table = _uninterrupted_table(num_problems, n, B)

    # Killed-then-resumed: pre-seed batches 0 and 2, resume the rest.
    fp = _fingerprint(B)
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    w = ManifestWriter(tmp_path / "resume" / "manifest.jsonl")
    # batch 0 -> problems 0,1 ; batch 2 -> problems 4,5
    for batch_idx, problems in ((0, (0, 1)), (2, (4, 5))):
        outputs = {str(p): [f"sol-{p}-{r}" for r in range(n)] for p in problems}
        w.append({"task": TASK, "batch_idx": batch_idx}, {"outputs": outputs})

    bench = _Bench(num_samples=n)
    bench.passk_batch_size = B
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    assert mgr.decide() == "resume"

    model = _FakeLM()
    resumed_out = bench.generate_n_samples_batched(model, _build_instances_factory(num_problems), n)

    # only batch 1 (problems 2,3) regenerated; 0/2 restored
    assert sorted(set(p for p, _ in model.generated)) == [2, 3]

    # per-problem outputs byte-identical to the uninterrupted run
    assert resumed_out == baseline_out

    # and the aggregated pass@k table is exactly identical
    num_correct = [sum(1 for s in per if int(s.rsplit("-", 1)[1]) % 2 == 0) for per in resumed_out]
    resumed_table = aggregate_pass_at_k(n, num_correct, [1, 2, 4])
    assert resumed_table == baseline_table
    assert set(resumed_table) == {"pass@1", "pass@2", "pass@4"}


def test_unit_key_and_payload_shape(tmp_path):
    """Units are {task, batch_idx}; payload carries per-(problem,sample) outputs."""
    num_problems, n, B = 4, 3, 2
    fp = _fingerprint(B)
    bench = _Bench(num_samples=n)
    bench.passk_batch_size = B
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    bench.generate_n_samples_batched(_FakeLM(), _build_instances_factory(num_problems), n)

    from eval.resume import read_manifest

    states = read_manifest(tmp_path / "resume" / "manifest.jsonl")
    keys = sorted(s.unit["batch_idx"] for s in states)
    assert keys == [0, 1]  # 4 problems / B=2 -> 2 batches
    assert all(s.unit["task"] == TASK for s in states)
    # batch 0 payload has problems 0,1 each with n completions
    b0 = next(s for s in states if s.unit["batch_idx"] == 0)
    assert set(b0.payload["outputs"]) == {"0", "1"}
    assert b0.payload["outputs"]["0"] == ["sol-0-0", "sol-0-1", "sol-0-2"]


def test_default_batch_size_is_single_batch(tmp_path):
    """No passk_batch_size set -> all problems in one batch (one resume unit)."""
    num_problems, n = 5, 2
    fp = _fingerprint()  # passk_batch_size=None
    bench = _Bench(num_samples=n)  # no passk_batch_size attr
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    bench.attach_resume_manager(mgr)
    bench.generate_n_samples_batched(_FakeLM(), _build_instances_factory(num_problems), n)

    from eval.resume import read_manifest

    states = read_manifest(tmp_path / "resume" / "manifest.jsonl")
    assert [s.unit["batch_idx"] for s in states] == [0]  # one batch
    assert set(states[0].payload["outputs"]) == {"0", "1", "2", "3", "4"}
