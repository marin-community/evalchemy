from eval.chat_benchmarks.hmmt_common import MathArenaHMMTBenchmark


class HMMTFeb2024Benchmark(MathArenaHMMTBenchmark):
    """HMMT February 2024 benchmark from MathArena."""

    DATASET_NAME = "MathArena/hmmt_feb_2024"
    EXPECTED_NUM_ROWS = 30
