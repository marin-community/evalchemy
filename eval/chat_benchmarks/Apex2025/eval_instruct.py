from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class Apex2025Benchmark(MathArenaFinalAnswerBenchmark):
    """Apex 2025 final-answer benchmark from MathArena."""

    DATASET_NAME = "MathArena/apex_2025"
    EXPECTED_NUM_ROWS = 12
    BENCHMARK_DESCRIPTION = "Apex 2025"
