from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class BRUMO2025Benchmark(MathArenaFinalAnswerBenchmark):
    """BRUMO 2025 final-answer benchmark from MathArena."""

    DATASET_NAME = "MathArena/brumo_2025"
    EXPECTED_NUM_ROWS = 30
    BENCHMARK_DESCRIPTION = "BRUMO 2025"
