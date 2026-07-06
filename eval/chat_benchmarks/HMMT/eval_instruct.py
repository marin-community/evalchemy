from eval.chat_benchmarks.hmmt_common import MathArenaHMMTBenchmark


class HMMTBenchmark(MathArenaHMMTBenchmark):
    """HMMT February 2025 benchmark from MathArena."""

    DATASET_NAME = "MathArena/hmmt_feb_2025"
    EXPECTED_NUM_ROWS = 30
