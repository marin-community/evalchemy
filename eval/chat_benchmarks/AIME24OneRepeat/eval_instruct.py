"""One-repeat AIME24 diagnostic using the canonical prompt and grader."""

from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark


class AIME24OneRepeatBenchmark(AIME24Benchmark):
    DEFAULT_REPEATS = 1
