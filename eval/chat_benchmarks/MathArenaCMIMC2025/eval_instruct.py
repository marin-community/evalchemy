from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class MathArenaCMIMC2025Benchmark(MathArenaFinalAnswerBenchmark):
    """CMIMC 2025 final-answer benchmark from MathArena."""

    DATASET_NAME = "MathArena/cmimc_2025"
    EXPECTED_NUM_ROWS = 40
    BENCHMARK_DESCRIPTION = "CMIMC 2025"
