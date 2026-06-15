"""Stage 5 — lm-eval ``_rank<R>.db`` interop edge hardening (CPU, lm_eval-gated).

Edge 2 from ``stage5_invalidation_hardening_scope.md``: a pre-existing legacy lm-eval cache
(``<use_cache>_rank<R>.db``) may have a different rank layout, be partially written / garbled,
or hold inconsistent entries. The manager must:
  * READ-only ADOPT the entries it can trust (greedy completions keyed the way ``CachingLM`` keys
    them) — proven in Stage 3b;
  * REFUSE / IGNORE what is inconsistent (corrupt db, non-string payloads, missing db) WITHOUT
    crashing and WITHOUT writing back to the SQLite db.

Requires ``lm_eval`` (for ``hash_args`` + the ``LM`` base) and ``sqlitedict`` — skipped on the
pure-stdlib otagent env; runs on abb.
"""

import pytest

pytest.importorskip("lm_eval")
pytest.importorskip("sqlitedict")

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.api.model import hash_args  # noqa: E402
from sqlitedict import SqliteDict  # noqa: E402

from eval.resume import ResumeManager, RunFingerprint  # noqa: E402
from eval.resume.lm_eval_native import _make_resume_caching_lm_cls  # noqa: E402


class _FakeLM:
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.generated = []

    def generate_until(self, requests, *a, **k):
        outs = []
        for req in requests:
            pid = getattr(req, "doc_id", None)
            self.generated.append(pid)
            outs.append(f"gen-{pid}")
        return outs


def _reqs(n):
    out = []
    for i in range(n):
        inst = Instance("generate_until", {"i": i}, (f"ctx-{i}", {"do_sample": False}), i)
        inst.doc_id = i
        out.append(inst)
    return out


def _fp():
    return RunFingerprint.from_run_inputs(model_repo="m", task_name="gsm8k", task_data_digest="sha256:x")


def _wrap(lm, mgr, interop_db=None):
    return _make_resume_caching_lm_cls()(lm, mgr, "gsm8k", interop_db=interop_db)


# --------------------------------------------------------------------------------------------
# corrupt / partially-written db -> ignored (no crash, no writeback)
# --------------------------------------------------------------------------------------------

def test_corrupt_db_file_is_ignored_no_crash(tmp_path):
    """A garbage (non-sqlite) file at the _rank<R>.db path is read-attempted then ignored."""
    db_path = tmp_path / "legacy_rank0.db"
    db_path.write_bytes(b"this is not a sqlite database \x00\x01\x02 garbage")

    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fp(), mode="auto")
    lm = _FakeLM()
    wrapped = _wrap(lm, mgr, interop_db=str(db_path))  # must NOT raise on construction

    out = wrapped.generate_until(_reqs(3))
    # nothing adopted from the unreadable db -> all 3 generated fresh
    assert out == ["gen-0", "gen-1", "gen-2"]
    assert [g for g in lm.generated] == [0, 1, 2]
    assert len(mgr.done_units()) == 3
    # we never wrote back to the SQLite db file (its bytes are unchanged garbage)
    assert db_path.read_bytes() == b"this is not a sqlite database \x00\x01\x02 garbage"


def test_inconsistent_db_entry_non_string_is_ignored(tmp_path):
    """A legacy entry whose payload is not a completion string is ignored (regenerated)."""
    reqs = _reqs(3)
    db_path = str(tmp_path / "legacy_rank0.db")
    with SqliteDict(db_path) as db:
        db[hash_args("generate_until", reqs[0].args)] = "legacy-0"  # valid
        db[hash_args("generate_until", reqs[1].args)] = {"unexpected": "schema"}  # inconsistent
        db[hash_args("generate_until", reqs[2].args)] = None  # half-written placeholder
        db.commit()

    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fp(), mode="auto")
    lm = _FakeLM()
    wrapped = _wrap(lm, mgr, interop_db=db_path)
    out = wrapped.generate_until(reqs)

    # problem 0 adopted from the db; 1 (dict) + 2 (None) ignored -> regenerated
    assert out[0] == "legacy-0"
    assert out[1] == "gen-1"
    assert out[2] == "gen-2"
    assert sorted(lm.generated) == [1, 2]
    # only the trustworthy entry was adopted into the manifest; the bad ones never recorded
    assert len(mgr.done_units()) == 3  # 1 adopted + 2 freshly generated/recorded


def test_legacy_db_with_extra_unrelated_keys_is_harmless(tmp_path):
    """A db holding keys that match NO current request (e.g. from a different rank layout) is
    simply not consulted -> those problems generate fresh; no crash, no false adoption."""
    reqs = _reqs(2)
    db_path = str(tmp_path / "legacy_rank0.db")
    with SqliteDict(db_path) as db:
        # keys for problems that are NOT in this run's request set (different layout/slice)
        db[hash_args("generate_until", (f"ctx-{99}", {"do_sample": False}))] = "legacy-99"
        db[hash_args("generate_until", (f"ctx-{100}", {"do_sample": False}))] = "legacy-100"
        db.commit()

    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fp(), mode="auto")
    lm = _FakeLM()
    wrapped = _wrap(lm, mgr, interop_db=db_path)
    out = wrapped.generate_until(reqs)
    # none of the current requests hit the (unrelated) cache -> both generated fresh
    assert out == ["gen-0", "gen-1"]
    assert sorted(lm.generated) == [0, 1]


def test_legacy_db_is_never_written_back(tmp_path):
    """READ-only interop: after a resume run the legacy db's contents are unchanged."""
    reqs = _reqs(2)
    db_path = str(tmp_path / "legacy_rank0.db")
    with SqliteDict(db_path) as db:
        db[hash_args("generate_until", reqs[0].args)] = "legacy-0"
        db.commit()

    with SqliteDict(db_path) as db:
        before = dict(db)

    mgr = ResumeManager(run_dir=tmp_path, fingerprint=_fp(), mode="auto")
    wrapped = _wrap(_FakeLM(), mgr, interop_db=db_path)
    wrapped.generate_until(reqs)

    with SqliteDict(db_path) as db:
        after = dict(db)
    assert after == before  # the manager owns resume now; the legacy db is read-only
