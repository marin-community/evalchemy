from types import SimpleNamespace
from pathlib import Path

import pytest
from lm_eval.api.instance import Instance

from eval.limits import parse_key_value_args, resolve_evaluation_limits
from eval.task import BaseBenchmark


def _args(**overrides):
    values = {
        "model_args": "pretrained=test/model,base_url=http://localhost/v1",
        "gen_kwargs": "temperature=0",
        "max_length": None,
        "max_tokens": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_limits_are_materialized_for_native_and_custom_paths():
    args = _args(max_length=32768, max_tokens=2048, gen_kwargs="temperature=0,max_new_tokens=2048")

    limits = resolve_evaluation_limits(args)

    assert limits.max_length == 32768
    assert limits.max_tokens == 2048
    assert parse_key_value_args(args.model_args)["max_length"] == "32768"
    native_kwargs = parse_key_value_args(args.gen_kwargs)
    assert native_kwargs == {"temperature": "0", "max_gen_toks": "2048"}
    assert args.max_tokens == 2048


@pytest.mark.parametrize("legacy_key", ["max_tokens", "max_new_tokens", "max_gen_toks"])
def test_legacy_output_aliases_resolve_to_one_contract(legacy_key):
    args = _args(model_args="pretrained=test/model,max_model_len=4096", gen_kwargs=f"{legacy_key}=512")

    limits = resolve_evaluation_limits(args)

    assert limits.max_length == 4096
    assert limits.max_tokens == 512
    assert parse_key_value_args(args.model_args)["max_length"] == "4096"
    assert parse_key_value_args(args.model_args)["max_model_len"] == "4096"
    assert parse_key_value_args(args.gen_kwargs) == {"max_gen_toks": "512"}


def test_conflicting_limit_spellings_fail_before_any_benchmark_runs():
    with pytest.raises(ValueError, match="conflicting max_tokens"):
        resolve_evaluation_limits(_args(max_tokens=512, gen_kwargs="max_gen_toks=1024"))

    with pytest.raises(ValueError, match="conflicting max_length"):
        resolve_evaluation_limits(_args(max_length=8192, model_args="pretrained=test/model,max_length=4096"))


class _BudgetBenchmark(BaseBenchmark):
    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, responses):
        raise NotImplementedError


class _FakeModel:
    world_size = 1
    rank = 0


def test_custom_benchmark_request_cannot_escape_resolved_output_cap():
    benchmark = _BudgetBenchmark()
    benchmark.set_evaluation_limits(max_length=32768, max_tokens=2048)
    instance = Instance(
        "generate_until",
        {},
        ("prompt", {"max_new_tokens": 17, "max_tokens": 19, "max_gen_toks": 23}),
        0,
    )

    normalized = benchmark._normalize_model_args(_FakeModel(), [instance])

    assert normalized[0].args[1]["max_new_tokens"] == 2048
    assert "max_tokens" not in normalized[0].args[1]
    assert "max_gen_toks" not in normalized[0].args[1]


def test_custom_prompt_budget_field_receives_the_same_context_limit():
    benchmark = _BudgetBenchmark()
    benchmark.max_model_length = 4096
    benchmark.max_new_tokens = 99

    benchmark.set_evaluation_limits(max_length=16384, max_tokens=1024)

    assert benchmark.max_model_length == 16384
    assert benchmark.max_new_tokens == 1024


def test_every_custom_benchmark_routes_generation_through_base_limit_guard():
    """All chat benchmarks must reach ``BaseBenchmark.compute`` before inference.

    That method owns the request-time output-cap override above.  A new
    benchmark which bypasses it would reintroduce a separate max-token path.
    """
    benchmarks_dir = Path(__file__).parents[1] / "eval" / "chat_benchmarks"
    missing_guard = [
        str(path.relative_to(benchmarks_dir))
        for path in sorted(benchmarks_dir.glob("*/eval_instruct.py"))
        if "self.compute(" not in path.read_text()
    ]

    assert not missing_guard, f"custom benchmarks bypassing the common limit guard: {missing_guard}"
