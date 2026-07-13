"""lm-eval-native (gsm8k) resume integration — supersede `--use_cache` (Stage 3b).

The lm-eval-native path (gsm8k via the plain `lm_eval` CLI / `simple_evaluate`,
`eval/eval.py`) is the ONLY path with any pre-existing resume: lm-eval's
`--use_cache` wraps the LM in `CachingLM` (`lm_eval/api/model.py:235`), which keys
a per-request SQLite cache on `hash_args("generate_until", req.args)`. That cache:

  * is **greedy-only** — it explicitly SKIPS caching any request whose
    `req.args[1].get("do_sample")` is truthy (`api/model.py:271-280`) so repeated
    sampled draws are not collapsed to one identical completion;
  * has **no sample index** in its key (the `do_sample` skip is its workaround);
  * is reachable only through `simple_evaluate`.

This module **supersedes** `--use_cache` with the unified `ResumeManager`
(decision #5): one interface (`should_skip`/`record`/`restore`/`finalize`) shared
with the chat_benchmark (3a) and pass@k (3c) paths, a per-problem completion
marker that **also works for `do_sample=True`** (the case `CachingLM` bypasses),
and a per-rank manifest that matches lm-eval's `_rank<R>.db` layout. The manager
still **READS** any pre-existing `_rank<R>.db` so a run that already has one keeps
its prior completions (interop, not removal).

Flag-off invariant (global #1): when no manager is attached, `resume_simple_evaluate`
calls upstream `simple_evaluate(**kwargs)` verbatim — the gsm8k path is byte-identical
to today and no `resume/` dir is written. The wrapper LM is constructed only when a
manager is active.

Unit key: `{task, problem_idx}` for greedy; `{task, problem_idx, sample_idx}` when the
request is sampled (`do_sample=True`) — fixing the n>1 key collision lm-eval has.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .manager import ResumeManager

logger = logging.getLogger(__name__)


def _request_unit_key(task_name: str, req: Any, sample_seen: Dict[Any, int]) -> Dict[str, Any]:
    """Build the per-request unit key.

    `req` is an lm-eval `Instance` whose `.doc_id` identifies the problem and whose
    `.args == (context, gen_kwargs)`. For a sampled request (`do_sample=True`) we add
    a `sample_idx` so the N draws of one problem are distinct units (the collision
    `CachingLM` works around by refusing to cache sampled requests). `doc_id` is the
    stable per-problem index lm-eval assigns; we fall back to `idx` if absent.
    """
    problem_idx = getattr(req, "doc_id", None)
    if problem_idx is None:
        problem_idx = getattr(req, "idx", None)
    unit: Dict[str, Any] = {"task": task_name, "problem_idx": problem_idx}

    gen_kwargs = req.args[1] if len(req.args) > 1 and isinstance(req.args[1], dict) else {}
    if gen_kwargs.get("do_sample", False):
        # Distinct draws of the same problem -> distinct units. We index by order of
        # appearance per problem so repeated identical requests map to 0, 1, 2, ...
        seen = sample_seen.get(problem_idx, 0)
        unit["sample_idx"] = seen
        sample_seen[problem_idx] = seen + 1
    return unit


def _make_resume_caching_lm_cls():
    """Build the `ResumeCachingLM` class as an `lm_eval.api.model.LM` subclass.

    Done lazily inside a factory so the module imports without `lm_eval` present
    (the resume package is pure-stdlib otherwise — manager/manifest/fingerprint are
    unit-testable on any Python). `simple_evaluate`'s `isinstance(model, LM)` gate
    (`evaluator.py:254`) REQUIRES the wrapped LM to be an `LM` subclass — `CachingLM`
    sidesteps this because lm-eval wraps it AFTER that check, but we pass our wrapper
    as a pre-initialized `model`, so it must pass the isinstance test.
    """
    from lm_eval.api.model import LM

    class ResumeCachingLM(LM):
        """A `CachingLM`-shaped LM wrapper backed by the unified `ResumeManager`.

        Mirrors `CachingLM`'s shape (`lm_eval/api/model.py:235`): `generate_until` is
        intercepted to (a) restore already-done problems from the manifest, (b)
        generate ONLY the remaining requests, (c) record each new completion;
        everything else delegates to the underlying LM. Unlike `CachingLM` it resumes
        **sampled** requests too (via the `sample_idx` unit-key component) and can read
        a pre-existing `_rank<R>.db` for interop.
        """

        def __init__(self, lm, manager, task_name, interop_db=None):
            super().__init__()
            self.lm = lm
            self._manager = manager
            self._task_name = task_name
            self._interop_db = interop_db
            self._interop_cache: Dict[str, Any] = {}
            manager.decide()  # refuse loudly on a material delta before any generation
            if interop_db:
                self._load_interop_db(interop_db)

        # rank / world_size delegate to the underlying LM (per-rank manifest layout).
        @property
        def rank(self):
            return getattr(self.lm, "rank", 0)

        @property
        def world_size(self):
            return getattr(self.lm, "world_size", 1)

        # loglikelihood paths are greedy-scored and out of gsm8k-generation scope —
        # delegate verbatim to the underlying LM (no resume layer).
        def loglikelihood(self, requests, *a, **k):
            return self.lm.loglikelihood(requests, *a, **k)

        def loglikelihood_rolling(self, requests, *a, **k):
            return self.lm.loglikelihood_rolling(requests, *a, **k)

        # The chat-template protocol members below are DEFINED on the base `LM`
        # class (`tokenizer_name` raises NotImplementedError; `chat_template` /
        # `apply_chat_template` return inert defaults), so Python resolves them on
        # the base class and `__getattr__` NEVER fires for them. Without these
        # explicit forwards, an `--apply_chat_template` run against an API model
        # (e.g. local-chat-completions) crashes on `wrapped.tokenizer_name` and
        # silently logs the wrong (empty) chat template. Forward them to the real
        # LM. `*args/**kwargs` keeps us resilient to lm-eval signature drift.
        @property
        def tokenizer_name(self):
            return self.lm.tokenizer_name

        def chat_template(self, *args, **kwargs):
            return self.lm.chat_template(*args, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            return self.lm.apply_chat_template(*args, **kwargs)

        def set_cache_hook(self, *args, **kwargs):
            return self.lm.set_cache_hook(*args, **kwargs)

        # pass any other attribute through to the underlying LM (tokenizer,
        # eot_token_id, tok_encode, etc.). Note: this only fires for names NOT
        # already defined on this class or the base `LM` (hence the explicit
        # forwards above for base-class-defined members).
        def __getattr__(self, attr):
            # Only reached for attributes not found on self/the class; delegate to the
            # underlying LM. Guard the bootstrap window before `self.lm` is set so an
            # early miss raises AttributeError (not a confusing KeyError).
            lm = self.__dict__.get("lm")
            if lm is None:
                raise AttributeError(attr)
            return getattr(lm, attr)

        _load_interop_db = _impl_load_interop_db
        _interop_lookup = _impl_interop_lookup
        generate_until = _impl_generate_until

    return ResumeCachingLM

# -- method implementations (module-level so the lazy LM subclass can adopt them) ----
def _impl_load_interop_db(self, path: str) -> None:
    """Load a legacy lm-eval `_rank<R>.db` so a run that already has one keeps its
    completions. READ-only — we never write back to the SQLite db (the manager owns
    resume now). Best-effort: a missing/unreadable db is simply ignored.
    """
    if not os.path.exists(path):
        return
    # Reject a non-SQLite / partially-written file by its header magic BEFORE handing it to
    # SqliteDict — SqliteDict opens a background writer thread that would raise
    # `sqlite3.DatabaseError: file is not a database` off-thread (uncatchable here) on garbage.
    # A valid SQLite file begins with the 16-byte magic "SQLite format 3\000".
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("resume[%s]: could not open legacy cache %s (%s); ignoring", self._task_name, path, exc)
        return
    if header != b"SQLite format 3\x00":
        logger.warning(
            "resume[%s]: legacy cache %s is not a valid SQLite db (bad header); ignoring (read-only interop)",
            self._task_name,
            path,
        )
        return
    try:
        from sqlitedict import SqliteDict  # lazy: only when a db is actually present

        with SqliteDict(path) as db:
            # CachingLM stores under hash_args("generate_until", req.args); we cannot
            # reverse the hash to a doc_id, so the interop cache is consulted by the
            # SAME per-request hash CachingLM would use (see _impl_interop_lookup).
            self._interop_cache = dict(db)
        logger.info(
            "resume[%s]: loaded %d entries from legacy lm-eval cache %s (interop, read-only)",
            self._task_name,
            len(self._interop_cache),
            path,
        )
    except Exception as exc:  # pragma: no cover - defensive interop
        logger.warning(
            "resume[%s]: could not read legacy lm-eval cache %s for interop (%s); ignoring",
            self._task_name,
            path,
            exc,
        )


def _impl_interop_lookup(self, req: Any) -> Optional[Any]:
    """Look a request up in the legacy `_rank<R>.db` by the SAME key CachingLM used.

    Returns the cached completion if present, else None. Sampled requests were never
    cached by CachingLM (it skips `do_sample=True`), so this only ever hits greedy.
    """
    if not self._interop_cache:
        return None
    try:
        from lm_eval.api.model import hash_args  # lazy import; lm_eval-only env

        hsh = hash_args("generate_until", req.args)
    except Exception:  # pragma: no cover - lm_eval shape drift
        return None
    cached = self._interop_cache.get(hsh)
    # Inconsistent / partially-written legacy entries (None, or a not-yet-resolved
    # placeholder) must NOT be adopted as a completed unit — treat them as a miss so the
    # request is regenerated rather than recorded as a corrupt "done" payload.
    if cached is None:
        return None
    # CachingLM stores the generate_until response, which is a plain completion string.
    # Anything else (a stray list/dict from a different cache schema) is not a trustworthy
    # completion -> ignore it (read-only interop, never crash, never writeback).
    if not isinstance(cached, str):
        logger.warning(
            "resume[%s]: legacy cache entry has unexpected type %s; ignoring (regenerating)",
            self._task_name,
            type(cached).__name__,
        )
        return None
    return cached


def _impl_generate_until(self, requests: List[Any], *args: Any, **kwargs: Any) -> List[str]:
    """Resume-aware `generate_until`.

    For each request: skip+restore if its unit is already in the manifest, else
    regenerate. Newly generated completions are recorded one-by-one. Works for both
    greedy (`do_sample=False`) and sampled (`do_sample=True`) — the latter is the
    case `CachingLM` bypasses entirely.
    """
    manager = self._manager
    results: List[Optional[str]] = [None] * len(requests)

    # Build a unit key per request (stable doc_id-based; sample_idx for sampled).
    sample_seen: Dict[Any, int] = {}
    keys = [_request_unit_key(self._task_name, req, sample_seen) for req in requests]

    restored = manager.restore()  # {canonical_key: payload}
    from .manifest import canonical_unit_key

    remaining_reqs: List[Any] = []
    remaining_positions: List[int] = []
    skipped = 0
    interop_hits = 0

    for pos, (req, unit) in enumerate(zip(requests, keys)):
        if manager.should_skip(unit):
            results[pos] = restored[canonical_unit_key(unit)]["output"]
            skipped += 1
            continue
        # Interop: a pre-existing lm-eval `_rank<R>.db` may already hold this
        # (greedy) completion; adopt it into the manifest so the manager owns it
        # going forward (decision #5 — read the legacy db, then supersede it).
        cached = self._interop_lookup(req)
        if cached is not None:
            results[pos] = cached
            manager.record(unit, {"output": cached})
            interop_hits += 1
            continue
        remaining_reqs.append(req)
        remaining_positions.append(pos)

    if skipped or interop_hits:
        logger.info(
            "resume[%s] rank=%s: skipped %d done units (%d from legacy _rank<R>.db), "
            "generating %d remaining",
            self._task_name,
            getattr(self.lm, "rank", 0),
            skipped + interop_hits,
            interop_hits,
            len(remaining_reqs),
        )

    if remaining_reqs:
        new_outputs = self.lm.generate_until(remaining_reqs, *args, **kwargs)
        for pos, req, output in zip(remaining_positions, remaining_reqs, new_outputs):
            results[pos] = output
            manager.record(keys[pos], {"output": output})

    manager.finalize()
    return results


def resume_simple_evaluate(simple_evaluate_fn, *, resume_manager_factory=None, **kwargs):
    """Thin wrapper around lm-eval `simple_evaluate` that wires the ResumeManager.

    Flag-off invariant (global #1): when ``resume_manager_factory`` is None (the
    default — nothing attaches it before Stage 4) this is **exactly**
    ``simple_evaluate_fn(**kwargs)``. The gsm8k path is byte-identical to today and
    no ``resume/`` dir is written. The wrapper LM is constructed only when a manager
    is active.

    When a factory is provided, it is called as ``factory(task_name) -> ResumeManager``
    (the manager already knows its run_dir / mode / fingerprint). We construct the LM
    the way upstream ``simple_evaluate`` does, wrap it in :class:`ResumeCachingLM`
    (passing the legacy ``<use_cache>_rank<R>.db`` for read-only interop), and hand
    the wrapped LM to upstream as a pre-initialized ``model`` object. This is a
    wrapper at the call site, NOT a rewrite of ``simple_evaluate``.

    Mode ``off``: the factory returns a manager with ``mode='off'`` (a pure no-op —
    never skips, never writes); we still pass the wrapped LM but its manager records
    nothing, so behavior matches today (the manager is inert).
    """
    if resume_manager_factory is None:
        # No manager — verbatim passthrough. Byte-identical to today (invariant #1).
        return simple_evaluate_fn(**kwargs)

    import lm_eval

    model = kwargs.pop("model")
    model_args = kwargs.get("model_args")
    batch_size = kwargs.get("batch_size")
    max_batch_size = kwargs.get("max_batch_size")
    device = kwargs.get("device")
    tasks = kwargs.get("tasks") or []
    use_cache = kwargs.pop("use_cache", None)

    # Construct the LM exactly as upstream simple_evaluate does when `model` is a str
    # (`lm_eval/evaluator.py:221-252`). If `model` is already an LM object, use it.
    if isinstance(model, str):
        init_args = {
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
            "device": device,
        }
        if isinstance(model_args, dict):
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(model_args, init_args)
        else:
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args or "", init_args
            )
    else:
        lm = model

    # gsm8k is a single task in the lm-eval-native path; name the unit by it.
    task_name = tasks[0] if tasks else "lm_eval"
    manager = resume_manager_factory(task_name)

    # Legacy lm-eval `_rank<R>.db` for READ-only interop (decision #5).
    interop_db = None
    if use_cache is not None:
        interop_db = use_cache + "_rank" + str(getattr(lm, "rank", 0)) + ".db"

    ResumeCachingLM = _make_resume_caching_lm_cls()
    wrapped = ResumeCachingLM(lm, manager, task_name, interop_db=interop_db)
    # Pass the wrapped LM as a pre-initialized model object; upstream will NOT re-wrap
    # in CachingLM because we drop `use_cache` (the manager supersedes it).
    return simple_evaluate_fn(model=wrapped, **kwargs)
