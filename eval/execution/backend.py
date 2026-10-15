"""Translate common model operations without adding a second dispatcher."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientSession, ClientTimeout, TCPConnector
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.api_models import TemplateAPI

from eval.completion_response import FailedGeneration
from eval.contracts.eval_spec import ModelRequest, RequestKind
from eval.limits import preflight_endpoint_generation
from eval.contracts.preflight import validate_model_request
from eval.runtime.pool import RetryPolicy, retry

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def retryable_transport(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in RETRYABLE_STATUSES
    return isinstance(error, (ClientConnectionError, ClientPayloadError, ConnectionError, TimeoutError))


class _Admitted:
    async def acquire(self) -> bool:
        return True

    def release(self) -> None:
        pass


@dataclass
class LMBackend:
    """One initialized LM, used as a transport or local inference implementation."""

    model: LM
    session: ClientSession | None = None
    _pending: list = field(default_factory=list, init=False, repr=False)
    _batch: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> LMBackend:
        if self.model.world_size != 1:
            raise ValueError("the shared runtime requires one client process; serve distributed models behind an endpoint")
        if isinstance(self.model, TemplateAPI):
            self.session = ClientSession(
                connector=TCPConnector(ssl=self.model.verify_certificate), timeout=ClientTimeout(total=None),
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._batch is not None:
            await self._batch
        if self.session is not None:
            await self.session.close()

    async def execute(self, request: ModelRequest) -> Any:
        validate_model_request(request)
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = self.model.apply_chat_template(prompt, **request.chat_template)
        parameters = dict(request.generation)
        if parameters.pop("do_sample", None) is False:
            parameters["temperature"] = 0.0
        if "max_new_tokens" in parameters:
            parameters["max_gen_toks"] = parameters.pop("max_new_tokens")
        if isinstance(self.model, TemplateAPI):
            return await self._api_request(request, prompt, parameters)
        args = (prompt, parameters)
        if request.kind is RequestKind.LIKELIHOOD:
            args = (prompt, request.continuation)
        elif request.kind is RequestKind.ROLLING_LIKELIHOOD:
            args = (prompt,)
        instance = Instance(request.kind.value, {}, args, 0, metadata=(
            request.metadata.get("task"), request.metadata.get("ordinal"), 1,
        ))
        instance.repeat_idx = request.metadata.get("trial") or 0
        future = asyncio.get_running_loop().create_future()
        self._pending.append((instance, future))
        if self._batch is None or self._batch.done():
            self._batch = asyncio.create_task(self._local_batches())
        return await future

    async def _local_batches(self):
        # Coalesce already admitted operations for local engines. The shared
        # pool bounds this queue; no source or second dispatcher admits work.
        await asyncio.sleep(0)
        while self._pending:
            pending, self._pending = self._pending, []
            for kind in dict.fromkeys(instance.request_type for instance, _ in pending):
                batch = [(instance, future) for instance, future in pending
                         if instance.request_type == kind and not future.cancelled()]
                if not batch:
                    continue
                try:
                    outputs = await asyncio.to_thread(getattr(self.model, kind), [item for item, _ in batch])
                    if len(outputs) != len(batch):
                        raise ValueError(f"model returned {len(outputs)} outputs for {len(batch)} requests")
                except Exception as error:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(error)
                else:
                    for (_, future), output in zip(batch, outputs, strict=True):
                        if not future.done():
                            future.set_result(output)

    async def _api_request(self, request: ModelRequest, prompt: Any, parameters: dict[str, Any]) -> Any:
        model = self.model
        if request.kind is RequestKind.GENERATE:
            bounded, _, cap = preflight_endpoint_generation(
                tokenizer=model.tokenizer,
                payloads=[prompt],
                gen_kwargs=parameters,
                context_length=request.context_length or (model.max_length + 1 if model.max_length is not None else None),
                chat_template_kwargs=getattr(model, "_evalchemy_chat_template_kwargs", None),
            )
            if bounded is not None:
                parameters = bounded
                if cap is not None:
                    parameters["_evalchemy_context_output_cap"] = cap
            return await self._call([prompt], parameters, generate=True)
        if request.kind is RequestKind.LIKELIHOOD:
            if prompt:
                context, continuation = model._encode_pair(prompt, request.continuation)
            else:
                context, continuation = [model.prefix_token_id], model.tok_encode(request.continuation)
            return await self._likelihood(context, continuation)
        windows = map(
            utils.make_disjoint_window,
            utils.get_rolling_token_windows(
                token_list=model.tok_encode(prompt), prefix_token=model.prefix_token_id,
                max_seq_len=model.max_length - 1, context_len=1,
            ),
        )
        total = 0.0
        for context, continuation in windows:
            score, _ = await self._likelihood(context, continuation)
            total += score
        return total

    async def _likelihood(self, context: list[int], continuation: list[int]) -> Any:
        tokens = (context + continuation)[-self.model.max_length:]
        context_length = len(context) - max(0, len(context) + len(continuation) - self.model.max_length)
        return await self._call([tokens], {}, generate=False, ctxlens=[context_length])

    async def _call(self, messages: list, parameters: dict, *, generate: bool, ctxlens=None) -> Any:
        model = self.model
        budget = getattr(model, "transport_retry_budget", None)
        policy = RetryPolicy(
            budget=float(model.timeout if budget is None else budget),
            attempt_timeout=float(getattr(model, "transport_attempt_timeout", model.timeout)),
            max_attempts=None if budget is not None else model.max_retries,
        )

        async def call():
            [output] = await model.amodel_call(
                session=self.session, sem=_Admitted(), messages=messages,
                generate=generate, gen_kwargs=parameters, ctxlens=ctxlens,
            )
            return output

        try:
            return await retry(call, policy, retryable_transport)
        except Exception as exc:
            if not generate or not retryable_transport(exc):
                raise
            category = "AgentTimeoutError" if isinstance(exc, TimeoutError) else "model_transport"
            return FailedGeneration(category)
