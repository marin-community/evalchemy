import importlib
import sys
import types

from eval.judges import JudgeConfig


class _PlainBenchmark:
    def __init__(self):
        self.constructed = True


class _JudgeBenchmark:
    def __init__(self, judge_config=None):
        self.judge_config = judge_config


def _install_lm_eval_stubs(monkeypatch):
    lm_eval = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    api = types.ModuleType("lm_eval.api")
    instance_mod = types.ModuleType("lm_eval.api.instance")
    model_mod = types.ModuleType("lm_eval.api.model")

    class Instance:
        pass

    class LM:
        pass

    class OpenAIChatCompletion:
        pass

    class OpenAICompletionsAPI:
        pass

    class VLLM:
        pass

    instance_mod.Instance = Instance
    model_mod.LM = LM
    models.openai_completions = types.SimpleNamespace(
        OpenAIChatCompletion=OpenAIChatCompletion,
        OpenAICompletionsAPI=OpenAICompletionsAPI,
    )
    models.vllm_causallms = types.SimpleNamespace(VLLM=VLLM)
    lm_eval.models = models
    lm_eval.api = api

    for name, module in {
        "lm_eval": lm_eval,
        "lm_eval.models": models,
        "lm_eval.api": api,
        "lm_eval.api.instance": instance_mod,
        "lm_eval.api.model": model_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_task_module(monkeypatch, request):
    old_task_module = sys.modules.pop("eval.task", None)
    _install_lm_eval_stubs(monkeypatch)
    task_module = importlib.import_module("eval.task")

    def cleanup():
        sys.modules.pop("eval.task", None)
        if old_task_module is not None:
            sys.modules["eval.task"] = old_task_module

    request.addfinalizer(cleanup)
    return task_module


def test_register_only_passes_judge_config_when_constructor_accepts_it(tmp_path, monkeypatch, request):
    task_module = _import_task_module(monkeypatch, request)
    config = JudgeConfig.from_model("gpt-5.5")

    plain = task_module.TaskManager(benchmarks_dir=str(tmp_path), judge_config=config)
    plain._register_benchmark("plain", _PlainBenchmark)

    judged = task_module.TaskManager(benchmarks_dir=str(tmp_path), judge_config=config)
    judged._register_benchmark("judged", _JudgeBenchmark)

    assert plain.benchmark_instances["plain"].constructed is True
    assert not hasattr(plain.benchmark_instances["plain"], "judge_config")
    assert judged.benchmark_instances["judged"].judge_config is config


def test_requires_judge_config_inspects_constructor_signature(tmp_path, monkeypatch, request):
    task_module = _import_task_module(monkeypatch, request)
    tm = task_module.TaskManager(benchmarks_dir=str(tmp_path))
    tm.tasks = {"plain": _PlainBenchmark, "judged": _JudgeBenchmark}

    assert tm.requires_judge_config("plain") is False
    assert tm.requires_judge_config("judged") is True
