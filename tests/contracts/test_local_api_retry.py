"""Transport retry behavior for Evalchemy's local API adapters."""

import asyncio
import json

import pytest
from aiohttp import ClientResponseError, web
from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr

from eval import robust_api
from eval.contracts.failures import FailureCategory
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.robust_api import capture_endpoint_failures
from eval.sample_logging import canonicalize_samples
from eval.serve_eval.local_api import RetryingLocalChatCompletions, RetryingLocalCompletions


class _Clock:
    def __init__(self):
        self.now = 0.0

    async def sleep(self, delay):
        self.now += delay
        await asyncio.sleep(0)


async def _endpoint(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


def _adapter(kind, base_url, budget):
    model_args = (
        f"model=served,base_url={base_url}/{kind},tokenizer_backend=none,"
        f"tokenized_requests=False,max_length=4096,timeout=5,transport_retry_budget={budget}"
    )
    if kind == "chat/completions":
        return RetryingLocalChatCompletions.create_from_arg_string(model_args)
    return RetryingLocalCompletions.create_from_arg_string(model_args)


def _prompt(kind):
    if kind == "chat/completions":
        return JsonChatStr(json.dumps([{"role": "user", "content": "question"}]))
    return "question"


def _response(kind):
    if kind == "chat/completions":
        return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "answer"}}]}
    return {"choices": [{"index": 0, "text": "answer"}]}


async def _generate(adapter, prompt):
    request = Instance("generate_until", {}, (prompt, {}), 0)
    return await asyncio.to_thread(adapter.generate_until, [request])


@pytest.mark.parametrize("kind", ["completions", "chat/completions"])
def test_local_api_recovers_after_ten_minute_proxy_outage(monkeypatch, kind):
    clock = _Clock()
    attempts = 0
    monkeypatch.setattr(robust_api, "monotonic", lambda: clock.now)
    monkeypatch.setattr(robust_api, "_transport_retry_sleep", clock.sleep)
    monkeypatch.setattr(robust_api.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if clock.now < 600:
            return web.json_response({"error": "proxy unavailable"}, status=502)
        return web.json_response(_response(kind))

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter(kind, base_url, 900)
            with capture_endpoint_failures() as failures:
                output = await _generate(adapter, _prompt(kind))
            return output, failures
        finally:
            await runner.cleanup()

    output, failures = asyncio.run(run())

    assert output == ["answer"]
    assert attempts > 8
    assert clock.now >= 600
    assert failures.counts == {}


@pytest.mark.parametrize("status", [408, 429, 503, 504])
def test_local_api_retries_other_transient_http_statuses(monkeypatch, status):
    clock = _Clock()
    attempts = 0
    monkeypatch.setattr(robust_api, "monotonic", lambda: clock.now)
    monkeypatch.setattr(robust_api, "_transport_retry_sleep", clock.sleep)
    monkeypatch.setattr(robust_api.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({"error": "unavailable"}, status=status)
        return web.json_response(_response("completions"))

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter("completions", base_url, 20)
            return await _generate(adapter, "question")
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == ["answer"]
    assert attempts == 2


@pytest.mark.parametrize("kind", ["completions", "chat/completions"])
def test_local_api_bad_request_fails_after_one_attempt(kind):
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return web.json_response({"error": "malformed request"}, status=400)

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter(kind, base_url, 900)
            with pytest.raises(ClientResponseError) as error:
                await _generate(adapter, _prompt(kind))
            assert error.value.status == 400
        finally:
            await runner.cleanup()

    asyncio.run(run())

    assert attempts == 1


def test_local_api_budget_exhaustion_remains_infrastructure_error(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(robust_api, "monotonic", lambda: clock.now)
    monkeypatch.setattr(robust_api, "_transport_retry_sleep", clock.sleep)
    monkeypatch.setattr(robust_api.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        return web.json_response({"error": "proxy unavailable"}, status=502)

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter("completions", base_url, 20)
            with capture_endpoint_failures() as failures:
                output = await _generate(adapter, "question")
            return output, failures
        finally:
            await runner.cleanup()

    output, failures = asyncio.run(run())
    record = canonicalize_samples("task", [{"resps": [output], "metrics": ["accuracy"], "accuracy": 0.0}])[0]
    sample = sample_from_lm_eval(
        "task",
        {
            **record,
            "doc_id": 0,
            "doc": {"question": "question"},
            "target": "answer",
            "arguments": [["question", {}]],
            "filtered_resps": [output[0]],
            "filter": "none",
        },
    )

    assert clock.now == 20
    assert failures.counts == {FailureCategory.MODEL_TRANSPORT: 1}
    assert record["failure_category"] == "model_transport"
    assert sample.output == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
