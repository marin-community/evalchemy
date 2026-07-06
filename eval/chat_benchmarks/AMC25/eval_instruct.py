from eval.chat_benchmarks.amc_common import AMCBenchmark


class AMC25Benchmark(AMCBenchmark):
    """AMC 12 2025 text-ready subset."""

    TASK_NAME = "AMC25"
    DATA_FILE = "eval/chat_benchmarks/AMC25/data/amc25.json"
