from eval.chat_benchmarks.hmmt_common import MathArenaHMMTBenchmark


class HMMTNov2025Benchmark(MathArenaHMMTBenchmark):
    """HMMT November 2025 benchmark from MathArena."""

    DATASET_NAME = "MathArena/hmmt_nov_2025"
    EXPECTED_NUM_ROWS = 30
