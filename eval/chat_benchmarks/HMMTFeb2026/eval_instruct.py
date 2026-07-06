from eval.chat_benchmarks.hmmt_common import MathArenaHMMTBenchmark


class HMMTFeb2026Benchmark(MathArenaHMMTBenchmark):
    """HMMT February 2026 benchmark from MathArena."""

    DATASET_NAME = "MathArena/hmmt_feb_2026"
    EXPECTED_NUM_ROWS = 33
