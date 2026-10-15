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


class Resources:
    def __init__(self, workers: int = 1):
        self.workers = min(workers, os.cpu_count() or 1)
        self.processes: ProcessPoolExecutor | None = None

    async def call(self, function: Callable[..., Result], args: tuple[Any, ...], mode: ExecutionMode) -> Result:
        if mode is ExecutionMode.THREAD_SAFE:
            return await asyncio.to_thread(function, *args)
        if mode is ExecutionMode.PROCESS_ISOLATED:
            if self.processes is None:
                self.processes = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
            return await asyncio.get_running_loop().run_in_executor(self.processes, function, *args)
        # Functions owning a sandbox may fork or use process-global signal state.
        # They run on the driver and own their child-process lifecycle.
        return function(*args)

    def close(self) -> None:
        if self.processes is not None:
            self.processes.shutdown(wait=True, cancel_futures=True)
            self.processes = None
