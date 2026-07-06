from eval.chat_benchmarks.amc_common import AMCBenchmark


class AMC24Benchmark(AMCBenchmark):
    """AMC 12 2024 text-ready subset."""

    TASK_NAME = "AMC24"
    DATA_FILE = "eval/chat_benchmarks/AMC24/data/amc24.json"
