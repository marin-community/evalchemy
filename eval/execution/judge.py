"""Judge transport using the same admission, retry and checkpoint lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from aiohttp import ClientSession

from eval.contracts.eval_spec import ModelRequest
from eval.execution.backend import retryable_transport, transport_retry_delay
from eval.graders.answer_equivalence import JudgeConfig
from eval.runtime.pool import RetryPolicy, retry
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice


@dataclass
class JudgeBackend:
    config: JudgeConfig
    session: ClientSession

    async def execute(self, request: ModelRequest) -> str:
        parameters = dict(request.generation)
        parameters.pop("do_sample", None)
        if "max_gen_toks" in parameters:
            parameters["max_tokens"] = parameters.pop("max_gen_toks")
        prompt = request.prompt
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

        async def call():
            async with self.session.post(
                self.config.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"model": self.config.model, "messages": messages, **parameters},
            ) as response:
                response.raise_for_status()
                document = await response.json()
                completion = completion_response_from_chat_choice(document, document["choices"][0])
                content = completion.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("judge returned an empty response")
                return CompletionText(content, completion, CompletionContentPolicy.FINAL_ONLY)

        return await retry(call, RetryPolicy(budget=180, attempt_timeout=60, max_attempts=3), retryable_transport, transport_retry_delay)
