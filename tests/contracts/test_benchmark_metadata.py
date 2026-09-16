from types import SimpleNamespace

from eval.contracts.benchmark_metadata import (
    MetricKind,
    SourceMetric,
    canonical_metric_name,
    canonicalize_results,
    describe_benchmarks,
    infer_benchmark_metadata,
    resolve_metric_metadata,
)
from eval.contracts.task_outcome import TaskRoute
from eval.task import BaseBenchmark


class _ChatBenchmark(BaseBenchmark):
    METRICS = ("em", "f1")
    PRIMARY_METRIC = "f1"

    def benchmark_size(self) -> int:
        return 120

    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError


class _LMEvalTask:
    eval_docs = tuple(range(50))

    def get_config(self, name):
        return {
            "metric_list": [
                {"metric": "exact_match", "higher_is_better": True},
                {"metric": "f1", "higher_is_better": True},
            ],
            "metadata": {"primary_metric": "f1"},
        }.get(name)


def test_metric_aliases_share_one_canonical_vocabulary():
    assert {canonical_metric_name(name) for name in ("acc", "accuracy_avg", "em", "exact_match")} == {
        "accuracy"
    }
    assert canonical_metric_name("pass@1") == "pass_at_1"


def test_equivalent_source_metrics_collapse_without_gating_the_task():
    metrics, primary = resolve_metric_metadata(
        [SourceMetric("acc"), SourceMetric("exact_match")],
        primary_metric="exact_match",
    )

    assert [metric.name for metric in metrics] == ["accuracy"]
    assert primary == "accuracy"


def test_metric_resolution_applies_primary_and_kind_overrides():
    metrics, primary = resolve_metric_metadata(
        [SourceMetric("judge_score")],
        primary_metric="judge_score",
        name_overrides={"judge_score": "accuracy"},
        kind_overrides={"judge_score": "binary"},
    )

    assert primary == "accuracy"
    assert metrics[0].source_name == "judge_score"
    assert metrics[0].kind is MetricKind.BINARY


def test_chat_metadata_tracks_native_pass_at_k_configuration():
    benchmark = _ChatBenchmark(num_samples=8, pass_at_k="1,8,32")

    description = benchmark.describe("chat")

    assert description is not None
    assert [metric.name for metric in description.metrics] == ["pass_at_1", "pass_at_8"]
    assert description.primary_metric == "pass_at_1"
    assert description.metric_kind is MetricKind.BINARY


def test_describe_benchmarks_hides_task_route_and_preserves_full_counts():
    chat = _ChatBenchmark()
    chat.set_evaluation_limits(limit=20)
    custom_manager = SimpleNamespace(get_benchmark=lambda name: chat)
    lm_eval_manager = SimpleNamespace(
        load_task_or_group=lambda names: {"group": {"leaf": _LMEvalTask()}},
    )

    descriptions = describe_benchmarks(
        ["chat", "group"],
        {"chat": TaskRoute.CUSTOM, "group": TaskRoute.LM_EVAL},
        custom_manager,
        lm_eval_manager,
        limit=10,
    )

    assert [description.to_dict() for description in descriptions] == [
        {
            "schema_version": 1,
            "task": "chat",
            "primary_metric": "f1",
            "metric_kind": "continuous",
            "metrics": [
                {
                    "name": "accuracy",
                    "source_name": "em",
                    "kind": "binary",
                    "higher_is_better": True,
                },
                {
                    "name": "f1",
                    "source_name": "f1",
                    "kind": "continuous",
                    "higher_is_better": True,
                },
            ],
            "n_benchmark": 120,
            "n_attempted": 20,
        },
        {
            "schema_version": 1,
            "task": "leaf",
            "primary_metric": "f1",
            "metric_kind": "continuous",
            "metrics": [
                {
                    "name": "accuracy",
                    "source_name": "exact_match",
                    "kind": "binary",
                    "higher_is_better": True,
                },
                {
                    "name": "f1",
                    "source_name": "f1",
                    "kind": "continuous",
                    "higher_is_better": True,
                },
            ],
            "n_benchmark": 50,
            "n_attempted": 10,
        },
    ]


def test_canonicalize_results_removes_source_aliases_and_filter_suffixes():
    descriptions = describe_benchmarks(
        ["group"],
        {"group": TaskRoute.LM_EVAL},
        SimpleNamespace(get_benchmark=lambda name: None),
        SimpleNamespace(load_task_or_group=lambda names: {"leaf": _LMEvalTask()}),
        limit=None,
    )

    results = canonicalize_results(
        {
            "leaf": {
                "exact_match,strict-match": 0.2,
                "exact_match,flexible-extract": 0.4,
                "f1,none": 0.6,
                "f1_stderr,none": 0.03,
            }
        },
        descriptions,
    )

    assert results == {"leaf": {"accuracy": 0.4, "f1": 0.6, "f1_stderr": 0.03}}


def test_completed_unknown_benchmark_uses_conservative_defaults():
    description = infer_benchmark_metadata(
        "external-task",
        {"judge_score": 0.75, "num_total": 40},
        n_attempted=40,
    )

    assert description.primary_metric == "judge_score"
    assert description.metric_kind is MetricKind.CONTINUOUS
    assert description.n_benchmark is None
    assert description.n_attempted == 40


def test_metadata_discovery_does_not_gate_undeclared_benchmarks():
    descriptions = describe_benchmarks(
        ["chat"],
        {"chat": TaskRoute.CUSTOM},
        SimpleNamespace(get_benchmark=lambda name: _ChatBenchmark()),
        SimpleNamespace(),
        limit=None,
    )
    assert descriptions

    benchmark = _ChatBenchmark()
    benchmark.METRICS = ()
    descriptions = describe_benchmarks(
        ["external-chat"],
        {"external-chat": TaskRoute.CUSTOM},
        SimpleNamespace(get_benchmark=lambda name: benchmark),
        SimpleNamespace(),
        limit=None,
    )
    assert descriptions == ()
