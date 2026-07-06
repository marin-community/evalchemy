from eval.chat_benchmarks.hmmt_common import MathArenaHMMTBenchmark


class HMMTFeb2023Benchmark(MathArenaHMMTBenchmark):
    """HMMT February 2023 benchmark from MathArena."""

    DATASET_NAME = "MathArena/hmmt_feb_2023"
    EXPECTED_NUM_ROWS = 30
