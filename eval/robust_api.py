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
import re
from collections import Counter

from eval.completion_response import (
    CompletionClassification,
    CompletionContentPolicy,
    CompletionText,
    completion_response_from_chat_choice,
)
from eval.limits import preflight_endpoint_generation

logger = logging.getLogger("eval.robust_api")

_PATCH_FLAG = "_marin_resilient_batch_patched"
_COMPLETION_PATCH_FLAG = "_marin_completion_normalization_patched"
_OPENAI_PAYLOAD_PATCH_FLAG = "_marin_openai_payload_patched"
_REQUEST_FAILURE_PREFIX = "[EVALCHEMY_INFRASTRUCTURE_ERROR]"
_MAX_REQUEST_FAILURE_DETAIL = 512
_OPENAI_FIXED_GENERATION_MODEL = re.compile(r"^(?:gpt-5|o[134])(?:$|[-.])", re.IGNORECASE)


def completion_response_quality_invalid(classifications: Counter[CompletionClassification]) -> bool:
    """Return whether a generation run has too many unusable chat completions."""
    total = sum(classifications.values())
    missing_final = sum(
        classifications[classification]
        for classification in (
            CompletionClassification.REASONING_ONLY,
            CompletionClassification.REASONING_ONLY_TRUNCATED,
            CompletionClassification.EMPTY,
        )
    )
    return total > 0 and missing_final / total >= 0.5


def request_failure_placeholder(exc: BaseException) -> str:
    """Return the artifact marker for an endpoint request that exhausted retries."""
    detail = " ".join(str(exc).split())
    if detail:
        detail = f": {detail[:_MAX_REQUEST_FAILURE_DETAIL]}"
    return f"{_REQUEST_FAILURE_PREFIX} {type(exc).__name__}{detail}"


def openai_model_requires_fixed_generation(model: object) -> bool:
    """Return whether an official OpenAI model rejects stops and temperature zero."""
    return isinstance(model, str) and bool(_OPENAI_FIXED_GENERATION_MODEL.match(model))


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


def apply_completion_normalization() -> bool:
    """Preserve reasoning content returned by lm-eval's chat-completions adapter."""
    try:
        from lm_eval.models.openai_completions import LocalChatCompletion
    except Exception as exc:  # noqa: BLE001 - never let the patch import break eval startup
        logger.warning("completion normalization: could not import lm-eval chat adapter (%r); patch skipped.", exc)
        return False

    if getattr(LocalChatCompletion, _COMPLETION_PATCH_FLAG, False):
        return True

    original_init = LocalChatCompletion.__init__
    original_generate_until = LocalChatCompletion.generate_until

    def __init__(self, *args, completion_content_policy: str = "combine", **kwargs):
        self.completion_content_policy = CompletionContentPolicy(completion_content_policy)
        self.completion_responses = []
        self.completion_response_summary = Counter()
        self.completion_response_quality_invalid = False
        original_init(self, *args, **kwargs)

    def parse_generations(self, outputs, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        generated = []
        for output in outputs:
            try:
                choices = output["choices"]
                parsed = [None] * len(choices)
                for choice in choices:
                    completion = completion_response_from_chat_choice(output, choice)
                    self.completion_responses.append(completion)
                    parsed[choice["index"]] = CompletionText(
                        completion.normalized_content(self.completion_content_policy),
                        completion,
                        self.completion_content_policy,
                    )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                # Preserve lm-eval's content-filter fallback for malformed choices.
                logger.warning("completion normalization: could not parse generation (%s)", exc)
                parsed = [""]
            generated.extend(parsed)
        return generated

    def generate_until(self, requests, *args, **kwargs):
        response_start = len(self.completion_responses)
        generated = original_generate_until(self, requests, *args, **kwargs)
        classifications = Counter(response.classification for response in self.completion_responses[response_start:])
        self.completion_response_summary.update(classifications)
        reasoning_only = sum(
            self.completion_response_summary[classification]
            for classification in (
                CompletionClassification.REASONING_ONLY,
                CompletionClassification.REASONING_ONLY_TRUNCATED,
            )
        )
        total = sum(self.completion_response_summary.values())
        self.completion_response_quality_invalid = completion_response_quality_invalid(self.completion_response_summary)
        if classifications and reasoning_only:
            logger.warning(
                "completion normalization: %d/%d responses used reasoning without final content (%s)",
                reasoning_only,
                total,
                dict(self.completion_response_summary),
            )
        if self.completion_response_quality_invalid:
            logger.error(
                "completion normalization: result quality is invalid because at least half of responses lacked final content"
            )
        return generated

    LocalChatCompletion.__init__ = __init__
    LocalChatCompletion.parse_generations = parse_generations
    LocalChatCompletion.generate_until = generate_until
    setattr(LocalChatCompletion, _COMPLETION_PATCH_FLAG, True)
    logger.info("completion normalization: patched lm-eval local-chat-completions.")
    return True


def apply_openai_payload_controls() -> bool:
    """Restrict OpenAI-specific generation controls to anchored OpenAI model names."""
    try:
        from lm_eval.models.openai_completions import OpenAIChatCompletion
        from lm_eval.models.utils import handle_stop_sequences
    except Exception as exc:  # noqa: BLE001 - never let the patch import break eval startup
        logger.warning("OpenAI payload controls: could not import lm-eval adapter (%r); patch skipped.", exc)
        return False

    if getattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, False):
        return True

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ):
        assert type(messages) is not str, "chat-completions require the --apply_chat_template flag."
        request_kwargs = dict(gen_kwargs or {})
        request_kwargs.pop("do_sample", False)
        max_tokens = request_kwargs.pop("max_tokens", request_kwargs.pop("max_gen_toks", self._max_gen_toks))
        temperature = request_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(request_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        payload = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **request_kwargs,
        }
        if openai_model_requires_fixed_generation(self.model):
            payload.pop("stop")
            payload["temperature"] = 1
        return payload

    OpenAIChatCompletion._create_payload = _create_payload
    setattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    logger.info("OpenAI payload controls: patched GPT-5 family matching.")
    return True


# Apply on import so `from eval import robust_api` is enough to activate the patch.
_APPLIED = apply()
_COMPLETION_NORMALIZATION_APPLIED = apply_completion_normalization()
_OPENAI_PAYLOAD_CONTROLS_APPLIED = apply_openai_payload_controls()
