"""Evalchemy's OpenAI-compatible endpoint adapters."""

from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


class _TransportRetryBudget:
    def __init__(self, *args, transport_retry_budget=900, transport_attempt_timeout=None, **kwargs):
        self.transport_retry_budget = float(transport_retry_budget)
        if self.transport_retry_budget <= 0:
            raise ValueError("transport_retry_budget must be positive")
        super().__init__(*args, **kwargs)
        self.transport_attempt_timeout = float(
            self.timeout if transport_attempt_timeout is None else transport_attempt_timeout
        )
        if self.transport_attempt_timeout <= 0:
            raise ValueError("transport_attempt_timeout must be positive")
        self.endpoint_concurrency = self._concurrent
        # lm-eval selects an unbounded synchronous retry path at _concurrent=1.
        # Select its async path and keep the HTTP limit at the requested value.
        self._concurrent = max(2, self._concurrent)


class RetryingLocalCompletions(_TransportRetryBudget, LocalCompletionsAPI):
    """Local completions with a wall-clock transport retry budget."""


class RetryingLocalChatCompletions(_TransportRetryBudget, LocalChatCompletion):
    """Local chat completions with a wall-clock transport retry budget."""
