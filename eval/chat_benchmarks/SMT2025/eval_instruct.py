from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class SMT2025Benchmark(MathArenaFinalAnswerBenchmark):
    """SMT 2025 final-answer benchmark from MathArena."""

    DATASET_NAME = "MathArena/smt_2025"
    EXPECTED_NUM_ROWS = 53
    BENCHMARK_DESCRIPTION = "SMT 2025"
