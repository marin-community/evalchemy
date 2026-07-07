from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class MathArenaApex2025Benchmark(MathArenaFinalAnswerBenchmark):
    """MathArena Apex 2025 final-answer benchmark."""

    DATASET_NAME = "MathArena/apex_2025"
    EXPECTED_NUM_ROWS = 12
    BENCHMARK_DESCRIPTION = "MathArena Apex 2025"
