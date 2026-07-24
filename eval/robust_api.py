# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Make lm-eval's async OpenAI-endpoint batch resilient to a single request error.

Upstream lm-eval ``TemplateAPI.get_batched_requests`` (``lm_eval/models/api_models.py``,
pinned here at v0.4.12) fires every request as a task and awaits them with
``tqdm_asyncio.gather(*tasks)`` with **no** ``return_exceptions``. Each task is wrapped
in a tenacity ``retry(..., reraise=True)``, so the moment ONE request still errors after
exhausting its retries, that exception propagates out of ``gather`` and **aborts the
entire eval batch** -- a single bad/slow/5xx request nukes the whole run. (In lm-eval
<= 0.4.9 the crash was additionally *masked* by an ``UnboundLocalError: outputs`` in the
``except`` logging path; v0.4.12 fixed the masking via ``locals().get('outputs', ...)``
but the batch-abort itself remains.)

This is exactly the failure that blocked the Grug 67B-A2B ``local-completions`` math
eval: the served model was healthy (a 200 payload-probe returned coherent math), yet one
transient async request error aborted the whole gsm8k gather.

The patch, applied as a monkeypatch so it stays a minimal, upstream-tracking delta,
prevents one exhausted request from aborting the batch. It returns a stable,
human-readable infrastructure-error marker instead of lm-eval's empty-generation
placeholder, so artifacts distinguish a request failure from a genuine empty model
completion. Retries themselves are unchanged (the tenacity wrapper is preserved
verbatim inside the guard).

Only the *generative* path (``generate=True`` -> ``generate_until``) is softened; the
loglikelihood path (``generate=False``) re-raises exactly as before, because turning a
failed logprob request into a placeholder would silently corrupt scoring.

Import for side effect (idempotent):

    from eval import robust_api  # noqa: F401  (patches lm-eval async-batch error handling)

``eval/eval.py`` does this at import, so every ``python -m eval.eval`` run -- and thus the
``eval.serve_eval`` runner that shells out to it -- gets the resilient batch.
"""

from __future__ import annotations

import logging

from eval.limits import preflight_endpoint_generation

logger = logging.getLogger("eval.robust_api")

_PATCH_FLAG = "_marin_resilient_batch_patched"
_REQUEST_FAILURE_PREFIX = "[EVALCHEMY_INFRASTRUCTURE_ERROR]"
_MAX_REQUEST_FAILURE_DETAIL = 512


def request_failure_placeholder(exc: BaseException) -> str:
    """Return the artifact marker for an endpoint request that exhausted retries."""
    detail = " ".join(str(exc).split())
    if detail:
        detail = f": {detail[:_MAX_REQUEST_FAILURE_DETAIL]}"
    return f"{_REQUEST_FAILURE_PREFIX} {type(exc).__name__}{detail}"


def apply() -> bool:
    """Patch ``TemplateAPI.get_batched_requests`` to be resilient. Idempotent.

    Returns True if the patch is (now) in place, False if it could not be applied
    (e.g. lm-eval drifted and the method/symbols are gone) -- in which case the
    unpatched upstream behavior is left untouched and a warning is logged.
    """
    try:
        import asyncio

        from aiohttp import ClientSession, ClientTimeout, TCPConnector
        from tenacity import retry, stop_after_attempt, wait_exponential
        from tqdm.asyncio import tqdm_asyncio

        from lm_eval.models import api_models as _api
        from lm_eval.models.utils import chunks
    except Exception as exc:  # noqa: BLE001 - never let the patch import break eval startup
        logger.warning("robust_api: could not import lm-eval async deps (%r); patch skipped.", exc)
        return False

    template_api = getattr(_api, "TemplateAPI", None)
    if template_api is None or not hasattr(template_api, "get_batched_requests"):
        logger.warning(
            "robust_api: lm_eval.models.api_models.TemplateAPI.get_batched_requests not found "
            "(lm-eval drifted from v0.4.12?); leaving upstream behavior unpatched."
        )
        return False

    if getattr(template_api, _PATCH_FLAG, False):
        return True  # already patched (idempotent across repeated imports)

    async def get_batched_requests(  # noqa: PLR0913 - mirrors the upstream signature verbatim
        self,
        requests,
        cache_keys,
        *,
        generate: bool = True,
        ctxlens=None,
        **kwargs,
    ):
        """Resilient mirror of lm-eval v0.4.12 ``TemplateAPI.get_batched_requests``.

        Identical to upstream except each per-request task is wrapped in a guard: a
        request that exhausts its retries returns an infrastructure-error marker
        rather than propagating out of ``gather`` and aborting the batch.
        """
        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        conn = TCPConnector(limit=self._concurrent, ssl=self.verify_certificate)
        sem = asyncio.Semaphore(self._concurrent)
        async with ClientSession(connector=conn, timeout=ClientTimeout(total=self.timeout)) as session:
            retry_ = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
                before_sleep=lambda retry_state: logger.info("Retry attempt %s", retry_state.attempt_number),
            )(self.amodel_call)

            async def _guarded(message, cache_key, ctxlen, call_kwargs):
                try:
                    return await retry_(
                        session=session,
                        sem=sem,
                        messages=message,
                        cache_keys=cache_key,
                        generate=generate,
                        ctxlens=ctxlen,
                        **call_kwargs,
                    )
                except BaseException as exc:  # noqa: BLE001 - one failed request must not nuke the batch
                    if not generate:
                        # Loglikelihood: a placeholder would corrupt scoring -> preserve
                        # upstream fail-fast behavior.
                        raise
                    n = len(message) if hasattr(message, "__len__") else 1
                    placeholder = request_failure_placeholder(exc)
                    logger.error(
                        "Request failed after all retries; returning an infrastructure-error "
                        "marker for %d prompt(s) (placeholder=%r). Cause: %r",
                        n,
                        placeholder,
                        exc,
                    )
                    # Cache the failure markers so a --use_cache resume does not re-issue them.
                    if cache_key:
                        for ck in cache_key:
                            self.cache_hook.add_partial("generate_until", ck, placeholder)
                    return [placeholder] * n

            tasks = []
            for message, cache_key, ctxlen in zip(
                chunks(requests, n=self._batch_size),
                chunks(cache_keys, n=self._batch_size),
                chunks(ctxlens, n=self._batch_size),
            ):
                request_kwargs = dict(kwargs)
                if generate:
                    # This is the one shared seam for lm-eval-native and every
                    # Evalchemy benchmark: the actual text/chat payload exists,
                    # but no HTTP request has been issued yet. ``max_length``
                    # on TemplateAPI is stored as context-1 by lm-eval.
                    try:
                        bounded, prompt_tokens, effective_cap = preflight_endpoint_generation(
                            tokenizer=self.tokenizer,
                            payloads=message,
                            gen_kwargs=kwargs.get("gen_kwargs"),
                            context_length=self.max_length + 1 if self.max_length is not None else None,
                        )
                    except Exception as exc:  # noqa: BLE001 - refuse deterministic overflow before transport
                        raise ValueError(f"endpoint context preflight failed: {exc}") from exc
                    if bounded is not None:
                        request_kwargs["gen_kwargs"] = bounded
                    if prompt_tokens is not None and effective_cap is not None:
                        logger.info(
                            "endpoint context preflight: largest_prompt=%d, max_output=%d, context=%d",
                            prompt_tokens,
                            effective_cap,
                            self.max_length + 1,
                        )
                tasks.append(asyncio.create_task(_guarded(message, cache_key, ctxlen, request_kwargs)))
            return await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    template_api.get_batched_requests = get_batched_requests
    setattr(template_api, _PATCH_FLAG, True)
    logger.info("robust_api: patched TemplateAPI.get_batched_requests (single-request errors leave markers).")
    return True


# Apply on import so `from eval import robust_api` is enough to activate the patch.
_APPLIED = apply()
