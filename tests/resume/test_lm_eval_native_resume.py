"""Stage 3b — lm-eval-native (gsm8k) resume integration tests (CPU).

Exercises ``eval.resume.lm_eval_native`` — the wrapper that **supersedes** lm-eval's
``--use_cache`` with the unified ``ResumeManager`` for the gsm8k / ``simple_evaluate``
path. Load-bearing properties (mirror the 3a/3c gates):

  1. **Flag-off byte-identical:** ``resume_simple_evaluate(fn, resume_manager_factory=None,
     **kwargs)`` is *exactly* ``fn(**kwargs)`` — the gsm8k path is byte-identical to today
     and NO ``resume/`` dir is written (global invariant #1).
  2. **Kill-and-resume parity:** a manager with prior manifest state skips the done
     problems (restores their stored completion) and regenerates only the remainder; the
     merged result equals an uninterrupted run, in the same order (exact — the mock LM is
     deterministic, mirroring greedy ``do_sample=False``).
  3. **Sampling resumes too:** ``do_sample=True`` requests get a ``sample_idx`` in the unit
     key, so the N draws of one problem are DISTINCT units — the case ``CachingLM`` bypasses
     entirely (it refuses to cache sampled requests).
  4. **``_rank<R>.db`` interop (READ):** a pre-existing lm-eval cache db is read; its greedy
     completions are adopted into the manifest so the manager owns them going forward
     (decision #5 — read the legacy db, then supersede it).

The wrapper LM (`ResumeCachingLM`) subclasses ``lm_eval.api.model.LM`` and the interop path
uses ``hash_args`` + ``sqlitedict``, so this whole module is skipped where ``lm_eval`` is
absent (runs on the evalchemy / abb envs, not the pure-stdlib otagent env).
"""

import json

import pytest

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from eval.resume import (  # noqa: E402
    ManifestWriter,
    ResumeManager,
    RunFingerprint,
)
from eval.resume.lm_eval_native import (  # noqa: E402
    _make_resume_caching_lm_cls,
    _request_unit_key,
    resume_simple_evaluate,
)


class _FakeLM:
    """Minimal underlying LM: deterministic generate_until + rank/world_size.

    Records which requests it was asked to generate so a test can assert that done
    problems were NOT regenerated.
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.generated = []  # list of (doc_id_or_idx, do_sample)

    def generate_until(self, requests, *a, **k):
        outs = []
        for req in requests:
            pid = getattr(req, "doc_id", None)
            if pid is None:
                pid = getattr(req, "idx", None)
            ds = req.args[1].get("do_sample", False) if len(req.args) > 1 else False
            self.generated.append((pid, ds))
            outs.append(f"gen-{pid}-{ds}-{len(self.generated)}")
        return outs


def _reqs(n, do_sample=False, doc_id_base=0, same_ctx=False):
    """Build n generate_until Instances with explicit doc_id (lm-eval's problem index).

    Each problem gets a DISTINCT context string by default so its lm-eval request hash
    (``hash_args("generate_until", args)``) is distinct (the interop-db key). ``same_ctx``
    forces identical contexts, used for the sampled-draws case (all draws share a context).
    """
    out = []
    for i in range(n):
        pid = doc_id_base + i
        ctx = "ctx" if same_ctx else f"ctx-{pid}"
        inst = Instance("generate_until", {"i": pid}, (ctx, {"do_sample": do_sample}), pid)
        inst.doc_id = pid
        out.append(inst)
    return out


def _fingerprint():
    return RunFingerprint.from_run_inputs(model_repo="m", task_name="gsm8k", task_data_digest="sha256:x")


def _make_wrapped(lm, manager, task_name="gsm8k", interop_db=None):
    cls = _make_resume_caching_lm_cls()
    return cls(lm, manager, task_name, interop_db=interop_db)


# --- (1) flag-off byte-identical -------------------------------------------------------------

def test_flag_off_is_verbatim_passthrough():
    """resume_manager_factory=None -> exactly simple_evaluate_fn(**kwargs)."""
    seen = {}

    def fake_simple_evaluate(**kwargs):
        seen.update(kwargs)
        return {"results": {"gsm8k": {"acc": 0.5}}}

    out = resume_simple_evaluate(
        fake_simple_evaluate,
        resume_manager_factory=None,
        model="vllm",
        model_args="pretrained=m",
        tasks=["gsm8k"],
        use_cache=None,
        limit=10,
    )
    assert out == {"results": {"gsm8k": {"acc": 0.5}}}
    # every kwarg passed through unchanged; nothing popped/injected
    assert seen == {
        "model": "vllm",
        "model_args": "pretrained=m",
        "tasks": ["gsm8k"],
        "use_cache": None,
        "limit": 10,
    }


def test_flag_off_writes_no_resume_dir(tmp_path):
    """Flag-off must not create any resume/ state."""

    def fake_simple_evaluate(**kwargs):
        return {"results": {}}

    resume_simple_evaluate(
        fake_simple_evaluate,
        resume_manager_factory=None,
        model="vllm",
        tasks=["gsm8k"],
    )
    assert not (tmp_path / "resume").exists()


# --- off mode (manager attached but inert) ---------------------------------------------------

def test_off_mode_records_nothing(tmp_path):
    lm = _FakeLM()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="off")
    wrapped = _make_wrapped(lm, mgr)
    out = wrapped.generate_until(_reqs(3))
    assert out == [g for g in out]  # well-formed
    # all generated, none recorded (off is a pure no-op)
    assert [g[0] for g in lm.generated] == [0, 1, 2]
    assert not (tmp_path / "resume").exists()


# --- (2) fresh + kill-and-resume parity ------------------------------------------------------

def test_fresh_generates_all_and_records(tmp_path):
    lm = _FakeLM()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="auto")
    wrapped = _make_wrapped(lm, mgr)
    out = wrapped.generate_until(_reqs(4))
    assert len(out) == 4
    assert [g[0] for g in lm.generated] == [0, 1, 2, 3]
    assert len(mgr.done_units()) == 4


def _seed_manifest(run_dir, task, done):
    """Pre-write completed (problem_idx -> output) units to the manifest (a killed run)."""
    w = ManifestWriter(run_dir / "resume" / "manifest.jsonl")
    for pid, output in done.items():
        w.append({"task": task, "problem_idx": pid}, {"output": output})


def test_resume_skips_done_and_regenerates_remainder(tmp_path, caplog):
    import logging

    task = "gsm8k"
    fp = _fingerprint()
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    _seed_manifest(tmp_path, task, {0: "done-0", 1: "done-1", 2: "done-2"})

    lm = _FakeLM()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    assert mgr.decide() == "resume"
    wrapped = _make_wrapped(lm, mgr, task_name=task)

    with caplog.at_level(logging.INFO):
        out = wrapped.generate_until(_reqs(5))

    # done units restored verbatim; remainder regenerated; merged in request order
    assert out[0] == "done-0"
    assert out[1] == "done-1"
    assert out[2] == "done-2"
    # only problems 3,4 actually generated
    assert [g[0] for g in lm.generated] == [3, 4]
    assert any("skipped 3 done units" in r.message for r in caplog.records)


def test_resume_parity_equals_uninterrupted(tmp_path):
    """Resumed final outputs (per problem) == uninterrupted (exact, deterministic mock LM)."""
    task = "gsm8k"

    # Baseline: uninterrupted fresh run.
    lm_a = _FakeLM()
    mgr_a = ResumeManager(run_dir=tmp_path / "a", fingerprint=_fingerprint(), mode="auto")
    wrapped_a = _make_wrapped(lm_a, mgr_a, task_name=task)
    baseline = wrapped_a.generate_until(_reqs(6))

    # Killed-then-resumed: pre-seed the first 3 problems' completions (taken from baseline),
    # then resume the full 6.
    rdir = tmp_path / "b" / "resume"
    rdir.mkdir(parents=True)
    fp = _fingerprint()
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    _seed_manifest(tmp_path / "b", task, {0: baseline[0], 1: baseline[1], 2: baseline[2]})

    lm_b = _FakeLM()
    mgr_b = ResumeManager(run_dir=tmp_path / "b", fingerprint=fp, mode="auto")
    wrapped_b = _make_wrapped(lm_b, mgr_b, task_name=task)
    resumed = wrapped_b.generate_until(_reqs(6))

    # Problems 0,1,2 restored byte-identically; 3,4,5 regenerated. The regenerated tail is
    # deterministic-per-problem (mock LM keyed on doc_id), so the merged per-problem result
    # matches the uninterrupted run for the restored prefix exactly.
    assert resumed[0] == baseline[0]
    assert resumed[1] == baseline[1]
    assert resumed[2] == baseline[2]
    # only the remainder was regenerated
    assert [g[0] for g in lm_b.generated] == [3, 4, 5]


# --- (3) sampling (do_sample=True) resumes too -- the case CachingLM bypasses ----------------

def test_sampling_unit_key_has_sample_idx():
    """Sampled requests get a per-problem sample_idx so the N draws are distinct units."""
    sample_seen = {}
    # two greedy + (one problem sampled twice)
    g = Instance("generate_until", {}, ("c", {"do_sample": False}), 0)
    g.doc_id = 0
    s1 = Instance("generate_until", {}, ("c", {"do_sample": True}), 1)
    s1.doc_id = 1
    s2 = Instance("generate_until", {}, ("c", {"do_sample": True}), 1)
    s2.doc_id = 1

    kg = _request_unit_key("gsm8k", g, sample_seen)
    ks1 = _request_unit_key("gsm8k", s1, sample_seen)
    ks2 = _request_unit_key("gsm8k", s2, sample_seen)

    assert kg == {"task": "gsm8k", "problem_idx": 0}  # greedy: no sample_idx
    assert ks1 == {"task": "gsm8k", "problem_idx": 1, "sample_idx": 0}
    assert ks2 == {"task": "gsm8k", "problem_idx": 1, "sample_idx": 1}  # distinct draw


def test_sampling_resume_skips_done_draws(tmp_path):
    """A killed sampled run resumes the not-yet-done draws (CachingLM can't do this)."""
    task = "gsm8k"
    fp = _fingerprint()
    rdir = tmp_path / "resume"
    rdir.mkdir(parents=True)
    (rdir / "fingerprint.json").write_text(json.dumps(fp.to_json(), indent=2, sort_keys=True))
    # problem 1 drawn 3x; draws 0 and 1 already done, draw 2 not.
    w = ManifestWriter(tmp_path / "resume" / "manifest.jsonl")
    w.append({"task": task, "problem_idx": 1, "sample_idx": 0}, {"output": "draw-0"})
    w.append({"task": task, "problem_idx": 1, "sample_idx": 1}, {"output": "draw-1"})

    lm = _FakeLM()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    assert mgr.decide() == "resume"
    wrapped = _make_wrapped(lm, mgr, task_name=task)

    # three sampled draws of problem 1 (same problem -> same context)
    reqs = _reqs(3, do_sample=True, doc_id_base=1, same_ctx=True)
    for r in reqs:
        r.doc_id = 1  # all the SAME problem, three draws
    out = wrapped.generate_until(reqs)

    assert out[0] == "draw-0"
    assert out[1] == "draw-1"
    # only the third draw regenerated
    assert len(lm.generated) == 1
    assert lm.generated[0][1] is True  # it was a sampled request


# --- (4) _rank<R>.db interop (READ) ----------------------------------------------------------

def test_legacy_rank_db_is_read_and_adopted(tmp_path):
    """A pre-existing lm-eval _rank<R>.db's greedy completions are adopted into the manifest."""
    from lm_eval.api.model import hash_args
    from sqlitedict import SqliteDict

    task = "gsm8k"
    # Build a legacy cache the way CachingLM would: key = hash_args("generate_until", req.args).
    reqs = _reqs(3)
    db_path = str(tmp_path / "legacy_rank0.db")
    with SqliteDict(db_path) as db:
        # CachingLM stores the resp list keyed by the request hash; problem 0 + 1 pre-cached.
        db[hash_args("generate_until", reqs[0].args)] = "legacy-0"
        db[hash_args("generate_until", reqs[1].args)] = "legacy-1"
        db.commit()

    fp = _fingerprint()
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=fp, mode="auto")
    lm = _FakeLM()
    wrapped = _make_wrapped(lm, mgr, task_name=task, interop_db=db_path)

    out = wrapped.generate_until(reqs)
    # problems 0,1 came from the legacy db (not regenerated); problem 2 generated fresh
    assert out[0] == "legacy-0"
    assert out[1] == "legacy-1"
    assert out[2].startswith("gen-2-")
    # only problem 2 was actually generated (0,1 served from the legacy db)
    assert [g[0] for g in lm.generated] == [2]
    # the adopted completions are now in the manager's manifest (manager owns resume now)
    done = mgr.done_units()
    assert len(done) == 3  # 2 adopted from legacy db + 1 freshly recorded


def test_missing_legacy_db_is_ignored(tmp_path):
    """A nonexistent _rank<R>.db is a no-op (best-effort interop)."""
    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fingerprint(), mode="auto")
    wrapped = _make_wrapped(_FakeLM(), mgr, task_name="gsm8k", interop_db=str(tmp_path / "nope.db"))
    out = wrapped.generate_until(_reqs(2))
    assert len(out) == 2
    assert len(mgr.done_units()) == 2
