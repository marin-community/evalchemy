"""Resource isolation for ordinary functions, independent of their callers."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from typing import Any, TypeVar

Result = TypeVar("Result")


class ExecutionMode(StrEnum):
    SERIAL = "serial"
    THREAD_SAFE = "thread_safe"
    PROCESS_ISOLATED = "process_isolated"
    SANDBOXED = "sandboxed"


def initialize_grader_worker() -> None:
    # The outer worker starts cleanly with spawn. Canonical code graders then
    # fork their local sandbox entrypoints from this worker's main thread.
    multiprocessing.set_start_method("fork", force=True)


class Resources:
    def __init__(self, workers: int = 1):
        self.workers = min(workers, os.cpu_count() or 1, 16)
        self.processes: ProcessPoolExecutor | None = None

    async def call(self, function: Callable[..., Result], args: tuple[Any, ...], mode: ExecutionMode) -> Result:
        if mode is ExecutionMode.THREAD_SAFE:
            return await asyncio.to_thread(function, *args)
        if mode in (ExecutionMode.PROCESS_ISOLATED, ExecutionMode.SANDBOXED):
            if self.processes is None:
                self.processes = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_grader_worker,
                )
            return await asyncio.get_running_loop().run_in_executor(self.processes, function, *args)
        return function(*args)

    def close(self) -> None:
        if self.processes is not None:
            self.processes.shutdown(wait=True, cancel_futures=True)
            self.processes = None
