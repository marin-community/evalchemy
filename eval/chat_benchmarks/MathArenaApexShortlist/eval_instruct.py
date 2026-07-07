from eval.chat_benchmarks.matharena_final_answer_common import MathArenaFinalAnswerBenchmark


class MathArenaApexShortlistBenchmark(MathArenaFinalAnswerBenchmark):
    """MathArena Apex Shortlist 2025 final-answer benchmark."""

    DATASET_NAME = "MathArena/apex-shortlist"
    EXPECTED_NUM_ROWS = 47
    BENCHMARK_DESCRIPTION = "MathArena Apex Shortlist 2025"
