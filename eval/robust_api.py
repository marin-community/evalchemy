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

The patch, applied as a monkeypatch so it stays a minimal, upstream-tracking delta,
prevents one exhausted generation request from cancelling its siblings. It supplies
an empty response with a classified failure, then lets scoring finish. Retries
remain unchanged. Loglikelihood requests still propagate errors because an empty
logprob cannot be scored.

Import for side effect (idempotent):

    from eval import robust_api  # noqa: F401  (patches lm-eval async-batch error handling)

``eval/eval.py`` does this at import, so every ``python -m eval.eval`` run -- and thus the
``eval.serve_eval`` runner that shells out to it -- gets the resilient batch.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator

from evalchemy_config.limits import MAX_OUTPUT_ALIASES

from eval.completion_response import (
    CompletionClassification,
    CompletionContentPolicy,
    CompletionResponse,
    CompletionText,
    FailedGeneration,
    MISSING_FINAL_CLASSIFICATIONS,
    completion_response_from_chat_choice,
)
from eval.contracts.failures import FailureCategory
from eval.generation_stops import bounded_request_stops
from eval.limits import ContextWindowExceededError, preflight_endpoint_generation

logger = logging.getLogger("eval.robust_api")

_PATCH_FLAG = "_marin_resilient_batch_patched"
_ROLLING_PATCH_FLAG = "_marin_rolling_batch_patched"
_COMPLETION_PATCH_FLAG = "_marin_completion_normalization_patched"
_OPENAI_PAYLOAD_PATCH_FLAG = "_marin_openai_payload_patched"
_GENERATION_OVERRIDES_ATTR = "_evalchemy_generation_overrides"
_ROLLING_WINDOWS_PER_CONCURRENT_SLOT = 4
_OPENAI_FIXED_GENERATION_MODEL = re.compile(r"^(?:gpt-5|o[134])(?:$|[-.])", re.IGNORECASE)
ENDPOINT_FAILURE_CATEGORIES = (
    FailureCategory.MODEL_TRANSPORT,
    FailureCategory.MALFORMED_MODEL_RESPONSE,
    FailureCategory.GRADER_INFRASTRUCTURE,
)
_active_failure_capture: ContextVar["EndpointFailureCapture | None"] = ContextVar(
    "evalchemy_endpoint_failure_capture",
    default=None,
)


@dataclass
class EndpointFailureCapture:
    """Task-scoped endpoint failures and response classes for result reporting."""

    counts: Counter[FailureCategory] = field(default_factory=Counter)
    response_summary: Counter[CompletionClassification] = field(default_factory=Counter)
    _lock: Lock = field(default_factory=Lock)

    def record(self, category: FailureCategory, count: int) -> None:
        if category not in ENDPOINT_FAILURE_CATEGORIES:
            raise ValueError(f"not an endpoint failure category: {category}")
        with self._lock:
            self.counts[category] += count

    def record_responses(self, classifications: Counter[CompletionClassification]) -> None:
        with self._lock:
            self.response_summary.update(classifications)


@contextmanager
def capture_endpoint_failures() -> Iterator[EndpointFailureCapture]:
    """Isolate endpoint diagnostics to one task invocation."""
    capture = EndpointFailureCapture()
    token = _active_failure_capture.set(capture)
    try:
        yield capture
    finally:
        _active_failure_capture.reset(token)


def record_endpoint_failure(category: FailureCategory, count: int = 1) -> None:
    """Record an endpoint failure for the task outcome."""
    capture = _active_failure_capture.get()
    if capture is not None:
        capture.record(category, count)


def record_completion_responses(classifications: Counter[CompletionClassification]) -> None:
    capture = _active_failure_capture.get()
    if capture is not None:
        capture.record_responses(classifications)


def completion_response_quality_invalid(classifications: Counter[CompletionClassification]) -> bool:
    """Return whether a generation run has too many unusable chat completions."""
    total = sum(classifications.values())
    missing_final = sum(
        classifications[classification]
        for classification in MISSING_FINAL_CLASSIFICATIONS
    )
    return total > 0 and missing_final / total >= 0.5


def openai_model_requires_fixed_generation(model: object) -> bool:
    """Return whether an official OpenAI model rejects stops and temperature zero."""
    return isinstance(model, str) and bool(_OPENAI_FIXED_GENERATION_MODEL.match(model))


def configure_generation_overrides(model: object, overrides: dict) -> None:
    """Give endpoint adapters the caller's generation settings, including an empty set."""
    # The runner imports this module without lm-eval installed. Import adapters
    # only when a model is being configured by the evaluation driver.
    from lm_eval.models.openai_completions import (
        LocalCompletionsAPI,
        OpenAIChatCompletion,
        OpenAICompletionsAPI,
    )

    if isinstance(model, LocalCompletionsAPI) and not isinstance(
        model, (OpenAICompletionsAPI, OpenAIChatCompletion)
    ):
        setattr(model, _GENERATION_OVERRIDES_ATTR, dict(overrides))


def parse_generation_overrides(value: str | dict | None) -> dict:
    """Parse caller settings and make temperature decisive for local decoders."""
    from lm_eval.utils import simple_parse_args_string

    overrides = simple_parse_args_string(value) if isinstance(value, str) else dict(value or {})
    if "temperature" in overrides and "do_sample" not in overrides:
        overrides["do_sample"] = float(overrides["temperature"]) > 0
    return overrides


def apply() -> bool:
    """Patch ``TemplateAPI.get_batched_requests`` to be resilient. Idempotent.

    Returns True if the patch is (now) in place, False if it could not be applied
    (e.g. lm-eval drifted and the method/symbols are gone) -- in which case the
    unpatched upstream behavior is left untouched and a warning is logged.
    """
    try:
        import asyncio

        from aiohttp import ClientSession, ClientTimeout, TCPConnector
        from lm_eval.models import api_models as _api
        from lm_eval.models.utils import chunks
        from tenacity import retry, stop_after_attempt, wait_exponential
        from tqdm.asyncio import tqdm_asyncio
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
        request that exhausts its retries returns an empty classified generation
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
                    record_endpoint_failure(FailureCategory.MODEL_TRANSPORT, n)
                    logger.error(
                        "Request failed after all retries; recording an empty generation for %d prompt(s). Cause: %r",
                        n,
                        exc,
                    )
                    return [FailedGeneration(FailureCategory.MODEL_TRANSPORT.value) for _ in range(n)]

            tasks = []
            skipped_preflight_logged = False
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
                    except ContextWindowExceededError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - add request-preflight context
                        raise ValueError(f"endpoint context preflight failed: {exc}") from exc
                    if bounded is not None:
                        request_kwargs["gen_kwargs"] = bounded
                    if (
                        self.tokenizer is None
                        and self.max_length is not None
                        and prompt_tokens is None
                        and bounded is not None
                        and any(alias in bounded for alias in MAX_OUTPUT_ALIASES)
                        and not skipped_preflight_logged
                    ):
                        logger.warning(
                            "endpoint context preflight skipped prompt-length check: no client tokenizer"
                        )
                        skipped_preflight_logged = True
                    if prompt_tokens is not None and effective_cap is not None:
                        logger.info(
                            "endpoint context preflight: largest_prompt=%d, max_output=%d, context=%d",
                            prompt_tokens,
                            effective_cap,
                            self.max_length + 1,
                        )
                tasks.append(asyncio.create_task(_guarded(message, cache_key, ctxlen, request_kwargs)))
            outputs = await tqdm_asyncio.gather(*tasks, desc="Requesting API")
            return outputs

    template_api.get_batched_requests = get_batched_requests
    setattr(template_api, _PATCH_FLAG, True)
    logger.info("robust_api: patched TemplateAPI.get_batched_requests (failed requests yield classified empty text).")
    return True


def apply_rolling_loglikelihood_batching() -> bool:
    """Batch rolling-likelihood windows across documents for API models."""
    try:
        from tqdm import tqdm

        from lm_eval import utils
        from lm_eval.models.api_models import TemplateAPI
    except Exception as exc:  # noqa: BLE001 - never let the patch import break eval startup
        logger.warning("rolling likelihood batching: could not import lm-eval symbols (%r); patch skipped.", exc)
        return False

    if getattr(TemplateAPI, _ROLLING_PATCH_FLAG, False):
        return True

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        loglikelihoods = []
        pending_documents = []
        pending_window_count = 0
        target_window_count = max(1, self._concurrent * _ROLLING_WINDOWS_PER_CONCURRENT_SLOT)

        def score_pending_documents():
            nonlocal pending_documents, pending_window_count
            if not pending_documents:
                return

            windows = [window for _, document_windows in pending_documents for window in document_windows]
            window_scores = self._loglikelihood_tokens(windows, disable_tqdm=True)
            offset = 0
            for string, document_windows in pending_documents:
                next_offset = offset + len(document_windows)
                string_nll = sum(score for score, _ in window_scores[offset:next_offset])
                loglikelihoods.append(string_nll)
                self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
                offset = next_offset

            pending_documents = []
            pending_window_count = 0

        for (string,) in tqdm([request.args for request in requests], disable=disable_tqdm):
            document_windows = [
                (None,) + window
                for window in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            ]
            pending_documents.append((string, document_windows))
            pending_window_count += len(document_windows)
            if pending_window_count >= target_window_count:
                score_pending_documents()

        score_pending_documents()
        return loglikelihoods

    TemplateAPI.loglikelihood_rolling = loglikelihood_rolling
    setattr(TemplateAPI, _ROLLING_PATCH_FLAG, True)
    logger.info("rolling likelihood batching: patched TemplateAPI.loglikelihood_rolling.")
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
        response_start = len(self.completion_responses)
        generated = []
        for output in outputs:
            try:
                choices = output["choices"]
                parsed = [None] * len(choices)
                parsed_responses = []
                for choice in choices:
                    completion = completion_response_from_chat_choice(output, choice)
                    parsed[choice["index"]] = CompletionText(
                        completion.normalized_content(self.completion_content_policy),
                        completion,
                        self.completion_content_policy,
                    )
                    parsed_responses.append(completion)
                self.completion_responses.extend(parsed_responses)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                record_endpoint_failure(FailureCategory.MALFORMED_MODEL_RESPONSE)
                logger.warning("completion normalization: could not parse generation (%s)", exc)
                completion = CompletionResponse(
                    content=None,
                    reasoning_content=None,
                    finish_reason=None,
                    usage=None,
                    provider_metadata={},
                    raw_choice={},
                    failure_category=FailureCategory.MALFORMED_MODEL_RESPONSE.value,
                )
                self.completion_responses.append(completion)
                parsed = [CompletionText("", completion, self.completion_content_policy)]
            generated.extend(parsed)
        parsed_responses = self.completion_responses[response_start:]
        classifications = Counter(response.classification for response in parsed_responses)
        record_completion_responses(classifications)
        missing_final = sum(
            classifications[classification]
            for classification in MISSING_FINAL_CLASSIFICATIONS
        )
        already_classified = sum(response.failure_category is not None for response in parsed_responses)
        if missing_final > already_classified:
            missing_count = missing_final - already_classified
            record_endpoint_failure(FailureCategory.MALFORMED_MODEL_RESPONSE, missing_count)
            logger.warning("completion normalization: %d responses lacked final content", missing_count)
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
            logger.warning(
                "completion normalization: at least half of responses lacked final content"
            )
        return generated

    LocalChatCompletion.__init__ = __init__
    LocalChatCompletion.parse_generations = parse_generations
    LocalChatCompletion.generate_until = generate_until
    setattr(LocalChatCompletion, _COMPLETION_PATCH_FLAG, True)
    logger.info("completion normalization: patched lm-eval local-chat-completions.")
    return True


def apply_openai_payload_controls() -> bool:
    """Bound API stops and restrict OpenAI-specific controls to OpenAI models."""
    try:
        from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion
        from lm_eval.models.utils import handle_stop_sequences
    except Exception as exc:  # noqa: BLE001 - never let the patch import break eval startup
        logger.warning("OpenAI payload controls: could not import lm-eval adapter (%r); patch skipped.", exc)
        return False

    if (
        getattr(LocalCompletionsAPI, _OPENAI_PAYLOAD_PATCH_FLAG, False)
        and getattr(LocalChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, False)
        and getattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, False)
    ):
        return True

    original_completions_payload = LocalCompletionsAPI._create_payload
    original_local_chat_payload = LocalChatCompletion._create_payload
    original_openai_chat_payload = OpenAIChatCompletion._create_payload

    def _bounded_generation_kwargs(gen_kwargs, eos, default_until=None):
        request_kwargs = dict(gen_kwargs or {})
        until = request_kwargs.get("until", default_until)
        stop = handle_stop_sequences(list(until) if isinstance(until, list) else until, eos)
        request_kwargs["until"] = bounded_request_stops(stop)
        return request_kwargs

    def _caller_generation_payload(self, payload):
        overrides = getattr(self, _GENERATION_OVERRIDES_ATTR, None)
        if overrides is None:
            return payload
        # The task YAML and lm-eval adapter both supply implicit sampling defaults.
        # Only caller settings should override the serving model's defaults.
        for key in ("temperature", "seed"):
            payload.pop(key, None)
        for key, value in overrides.items():
            if key == "max_gen_toks":
                token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                payload[token_key] = value
            elif key not in ("do_sample", "until"):
                payload[key] = value
        if overrides.get("do_sample") is False and "temperature" not in overrides:
            payload["temperature"] = 0
        return payload

    def _create_local_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos=None,
        **kwargs,
    ):
        payload = original_local_chat_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=_bounded_generation_kwargs(gen_kwargs, eos),
            seed=seed,
            eos=None,
            **kwargs,
        )
        return _caller_generation_payload(self, payload)

    def _create_completions_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos=None,
        **kwargs,
    ):
        if not generate:
            payload = original_completions_payload(
                self,
                messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=seed,
                eos=eos,
                **kwargs,
            )
            return _caller_generation_payload(self, payload)
        payload = original_completions_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=_bounded_generation_kwargs(gen_kwargs, eos),
            seed=seed,
            eos=None,
            **kwargs,
        )
        return _caller_generation_payload(self, payload)

    def _create_openai_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ):
        request_kwargs = _bounded_generation_kwargs(gen_kwargs, eos, ["<|endoftext|>"])
        selected_stops = list(request_kwargs["until"])
        temperature = request_kwargs.get("temperature", 0)
        payload = original_openai_chat_payload(
            self,
            messages,
            generate=generate,
            gen_kwargs=request_kwargs,
            seed=seed,
            eos=None,
            **kwargs,
        )
        if openai_model_requires_fixed_generation(self.model):
            payload.pop("stop", None)
            payload["temperature"] = 1
        else:
            payload["stop"] = selected_stops
            payload["temperature"] = temperature
        return _caller_generation_payload(self, payload)

    LocalCompletionsAPI._create_payload = _create_completions_payload
    LocalChatCompletion._create_payload = _create_local_payload
    OpenAIChatCompletion._create_payload = _create_openai_payload
    setattr(LocalCompletionsAPI, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    setattr(LocalChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    setattr(OpenAIChatCompletion, _OPENAI_PAYLOAD_PATCH_FLAG, True)
    logger.info("OpenAI payload controls: patched bounded stop selection and GPT-5 family matching.")
    return True


# Apply on import so `from eval import robust_api` is enough to activate the patch.
_APPLIED = apply()
_ROLLING_LOGLIKELIHOOD_BATCHING_APPLIED = apply_rolling_loglikelihood_batching()
_COMPLETION_NORMALIZATION_APPLIED = apply_completion_normalization()
_OPENAI_PAYLOAD_CONTROLS_APPLIED = apply_openai_payload_controls()
