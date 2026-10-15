"""A bounded worker pool with deterministic lookahead and request retries."""

from __future__ import annotations

import asyncio
from asyncio import sleep
import logging
import random
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from time import monotonic
from typing import Generic, TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Bound a transient operation by attempts, elapsed time and attempt timeout."""

    budget: float
    attempt_timeout: float
    max_attempts: int | None = None


async def retry(
    call: Callable[[], Awaitable[Result]],
    policy: RetryPolicy,
    retryable: Callable[[Exception], bool],
) -> Result:
    deadline = monotonic() + policy.budget
    attempts = 0
    async with asyncio.timeout(policy.budget):
        while True:
            attempts += 1
            try:
                async with asyncio.timeout(policy.attempt_timeout):
                    return await call()
            except Exception as exc:
                if not retryable(exc) or (policy.max_attempts is not None and attempts >= policy.max_attempts):
                    raise
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise
                delay = min(random.uniform(0, min(2 ** min(attempts - 1, 4), 10)), remaining)
                logger.info("Retrying request after attempt %d in %.1f seconds: %r", attempts, delay, exc)
                await sleep(delay)


@dataclass(frozen=True)
class Completed(Generic[Job, Result]):
    position: int
    job: Job
    result: Result | None = None
    error: Exception | None = None


class ExecutionPool:
    """Keep a fixed number of workers and a bounded window of source items.

    Results arrive in completion order. Positions preserve source order without
    retaining an entire dataset. The caller owns persistence and error policy.
    """

    def __init__(self, workers: int, lookahead: int | None = None):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.lookahead = max(workers, lookahead if lookahead is not None else workers * 4)

    async def map(
        self,
        jobs: Iterable[Job],
        execute: Callable[[Job], Awaitable[Result]],
        *,
        group: Callable[[Job], tuple] = lambda _job: (),
    ) -> AsyncIterator[Completed[Job, Result]]:
        source = enumerate(jobs)
        ready: deque[tuple[int, Job]] = deque()

        def next_job() -> tuple[int, Job] | None:
            if not ready:
                ready.extend(sorted(islice(source, self.lookahead), key=lambda item: group(item[1])))
            return ready.popleft() if ready else None

        pending: dict[asyncio.Task[Result], tuple[int, Job]] = {}
        exhausted = False
        count = failures = 0
        last_report = monotonic()
        try:
            while pending or not exhausted:
                while len(pending) < self.workers and not exhausted:
                    item = next_job()
                    if item is None:
                        exhausted = True
                    else:
                        pending[asyncio.create_task(execute(item[1]))] = item
                if not pending:
                    break
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    position, job = pending.pop(task)
                    try:
                        completed = Completed(position, job, result=task.result())
                    except Exception as exc:
                        completed = Completed(position, job, error=exc)
                        failures += 1
                    count += 1
                    if monotonic() - last_report >= 10:
                        logger.info("Pool progress: completed=%d failed=%d pending=%d", count, failures, len(pending))
                        last_report = monotonic()
                    yield completed
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info("Pool finished: completed=%d failed=%d", count, failures)
