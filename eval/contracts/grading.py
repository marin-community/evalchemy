"""Resource isolation required by a registered single-item grader."""

from enum import StrEnum


class GraderExecutionMode(StrEnum):
    SERIAL = "serial"
    THREAD_SAFE = "thread_safe"
    PROCESS_ISOLATED = "process_isolated"
    SANDBOXED = "sandboxed"
