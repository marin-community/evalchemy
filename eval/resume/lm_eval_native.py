"""Resume lm-eval requests from committed FineStore rows.

The sample manifest assigns stable unit keys, including distinct keys for
repeated sampled requests. The wrapped LM restores completed requests before
calling the model and commits new responses through the FineStore manager.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from eval.robust_api import configure_generation_overrides, parse_generation_overrides
from eval.contracts.sample_manifest import (
    DEFAULT_SAMPLE_NAMESPACE,
    SampleEntry,
    SampleIdentityError,
    SampleManifest,
    SampleRequest,
    canonical_json_identity,
)


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


def _plan_request_batch(
    manifest: SampleManifest, requests: List[Any]
) -> tuple[List[SampleEntry | None], tuple[SampleEntry, ...]]:
    """Map lm-eval request clones to stable sample units before an LM call.

    Multiple-choice requests share a document identity, while repeated sampling
    clones the same ``Instance`` object. Distributed padding may clone it beyond
    its declared repeat count; those padding calls are deliberately not samples.
    """
    occurrences: Dict[int, int] = {}
    sampled_occurrences: Dict[tuple[str, str], int] = {}
    ordinals: Dict[tuple[str, str], int] = {}
    planned: Dict[tuple[str, str, int | None], SampleRequest] = {}
    request_keys: List[tuple[str, str, int | None] | None] = []

    for request in requests:
        source_id = getattr(request, "doc_id", None)
        namespace = getattr(request, "task_name", None) or DEFAULT_SAMPLE_NAMESPACE
        try:
            encoded_source = canonical_json_identity(source_id)
        except SampleIdentityError as exc:
            raise SampleIdentityError(
                f"{manifest.task_name}: lm-eval sample source_id {exc}"
            ) from exc

        object_key = id(request)
        object_occurrence = occurrences.get(object_key, 0)
        occurrences[object_key] = object_occurrence + 1
        repeats = getattr(request, "repeats", None) or 1
        if object_occurrence >= repeats:
            request_keys.append(None)
            continue
        source_key = (namespace, encoded_source)
        ordinal = ordinals.setdefault(source_key, len(ordinals))
        gen_kwargs = request.args[1] if len(request.args) > 1 and isinstance(request.args[1], dict) else {}
        sampled = bool(gen_kwargs.get("do_sample", False))
        if sampled:
            repeat = sampled_occurrences.get(source_key, 0)
            sampled_occurrences[source_key] = repeat + 1
        elif repeats > 1:
            repeat = object_occurrence
        else:
            repeat = None
        unit_key = (namespace, encoded_source, repeat)
        planned.setdefault(
            unit_key,
            SampleRequest(
                source_id=source_id,
                ordinal=ordinal,
                namespace=namespace,
                shard=None,
                repeat=repeat,
            ),
        )
        request_keys.append(unit_key)

    entries = manifest.plan_batch(list(planned.values()))
    by_key = dict(zip(planned, entries))
    return [by_key[key] if key is not None else None for key in request_keys], entries


def _track_request_method(self, method_name: str, requests: List[Any], *args: Any, **kwargs: Any):
    request_entries, unique_entries = _plan_request_batch(self._sample_manifest, requests)
    outputs = getattr(self.lm, method_name)(requests, *args, **kwargs)
    self._sample_manifest.validate_output_count(len(requests), outputs)
    self._sample_manifest.mark_generated(unique_entries, [None] * len(unique_entries))
    return outputs


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
        """An LM wrapper that restores completed FineStore requests.

        Mirrors `CachingLM`'s shape (`lm_eval/api/model.py:235`): `generate_until` is
        intercepted to (a) restore already-done problems from the manifest, (b)
        generate ONLY the remaining requests, (c) record each new completion;
        everything else delegates to the underlying LM. Sampled requests use
        distinct unit keys, so each draw can resume independently.
        """

        def __init__(self, lm, manager, task_name, sample_manifest=None):
            super().__init__()
            self.lm = lm
            self._manager = manager
            self._task_name = task_name
            self._sample_manifest = sample_manifest or SampleManifest(task_name)
            if manager is not None:
                manager.decide()  # refuse loudly on a material delta before any generation

        # Rank and world size determine the FineStore resume namespace.
        @property
        def rank(self):
            return getattr(self.lm, "rank", 0)

        @property
        def world_size(self):
            return getattr(self.lm, "world_size", 1)

        # Likelihood requests are not resumable yet, but still cross the shared
        # identity and coverage boundary.
        def loglikelihood(self, requests, *a, **k):
            return _track_request_method(self, "loglikelihood", requests, *a, **k)

        def loglikelihood_rolling(self, requests, *a, **k):
            return _track_request_method(self, "loglikelihood_rolling", requests, *a, **k)

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

        generate_until = _impl_generate_until

    return ResumeCachingLM

# -- method implementations (module-level so the lazy LM subclass can adopt them) ----
def _impl_generate_until(self, requests: List[Any], *args: Any, **kwargs: Any) -> List[str]:
    """Resume-aware `generate_until`.

    For each request: skip+restore if its unit is already in the manifest, else
    regenerate. Newly generated completions are recorded one-by-one. Works for both
    greedy (`do_sample=False`) and sampled (`do_sample=True`) — the latter is the
    case `CachingLM` bypasses entirely.
    """
    manager = self._manager
    request_entries, unique_entries = _plan_request_batch(self._sample_manifest, requests)
    if manager is None:
        outputs = self.lm.generate_until(requests, *args, **kwargs)
        self._sample_manifest.validate_output_count(len(requests), outputs)
        self._sample_manifest.mark_generated(unique_entries, [None] * len(unique_entries))
        return outputs
    results: List[Optional[str]] = [None] * len(requests)

    # Build a unit key per request (stable doc_id-based; sample_idx for sampled).
    keys = [
        entry.resume_unit(self._task_name) if entry is not None else _request_unit_key(self._task_name, req, {})
        for req, entry in zip(requests, request_entries)
    ]
    restored = manager.restore()  # {canonical_key: payload}
    self._sample_manifest.validate_prior_entries(list(restored.values()), unique_entries)
    from .unit_keys import find_restored_payload

    remaining_reqs: List[Any] = []
    remaining_positions: List[int] = []
    skipped = 0
    for pos, (req, unit) in enumerate(zip(requests, keys)):
        restored_match = find_restored_payload(restored, [unit])
        if restored_match is not None:
            _, payload = restored_match
            results[pos] = payload["output"]
            skipped += 1
            continue
        remaining_reqs.append(req)
        remaining_positions.append(pos)

    if skipped:
        logger.info(
            "resume[%s] rank=%s: skipped %d done units, generating %d remaining",
            self._task_name,
            getattr(self.lm, "rank", 0),
            skipped,
            len(remaining_reqs),
        )

    if remaining_reqs:
        new_outputs = self.lm.generate_until(remaining_reqs, *args, **kwargs)
        self._sample_manifest.validate_output_count(len(remaining_reqs), new_outputs)
        for pos, req, output in zip(remaining_positions, remaining_reqs, new_outputs):
            results[pos] = output
            if request_entries[pos] is not None:
                manager.record(
                    keys[pos],
                    {"output": output, "sample": request_entries[pos].to_dict()},
                )

    manager.finalize()
    self._sample_manifest.mark_generated(unique_entries, [None] * len(unique_entries))
    return results


def resume_simple_evaluate(
    simple_evaluate_fn,
    *,
    resume_manager_factory=None,
    sample_manifest: SampleManifest | None = None,
    **kwargs,
):
    """Wrap lm-eval's model with FineStore request restoration and sample accounting."""
    if resume_manager_factory is None and sample_manifest is None:
        return simple_evaluate_fn(**kwargs)

    import lm_eval

    model = kwargs.pop("model")
    model_args = kwargs.get("model_args")
    batch_size = kwargs.get("batch_size")
    max_batch_size = kwargs.get("max_batch_size")
    device = kwargs.get("device")
    tasks = kwargs.get("tasks") or []

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

    overrides = parse_generation_overrides(kwargs.get("gen_kwargs"))
    if overrides:
        kwargs["gen_kwargs"] = overrides
    configure_generation_overrides(lm, overrides)

    # gsm8k is a single task in the lm-eval-native path; name the unit by it.
    task_name = tasks[0] if tasks else "lm_eval"
    manager = resume_manager_factory(task_name) if resume_manager_factory is not None else None
    sample_manifest = sample_manifest or SampleManifest(task_name)

    ResumeCachingLM = _make_resume_caching_lm_cls()
    wrapped = ResumeCachingLM(
        lm,
        manager,
        task_name,
        sample_manifest,
    )
    # Upstream accepts the pre-initialized model without adding its own cache.
    return simple_evaluate_fn(model=wrapped, **kwargs)
