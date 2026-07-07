from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class ApexShortlistBenchmark(MathArenaFinalAnswerBenchmark):
    """Apex Shortlist 2025 final-answer benchmark from MathArena."""

    DATASET_NAME = "MathArena/apex-shortlist"
    EXPECTED_NUM_ROWS = 47
    BENCHMARK_DESCRIPTION = "Apex Shortlist 2025"
