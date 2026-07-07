import ast
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


TASK_SPECS = [
    {
        "task_name": "Apex2025",
        "class_name": "Apex2025Benchmark",
        "dataset_name": "MathArena/apex_2025",
        "expected_rows": 12,
        "config_file": "matharena_apex_2025.yaml",
    },
    {
        "task_name": "ApexShortlist",
        "class_name": "ApexShortlistBenchmark",
        "dataset_name": "MathArena/apex-shortlist",
        "expected_rows": 47,
        "config_file": "matharena_apex_shortlist.yaml",
    },
    {
        "task_name": "CMIMC2025",
        "class_name": "CMIMC2025Benchmark",
        "dataset_name": "MathArena/cmimc_2025",
        "expected_rows": 40,
        "config_file": "matharena_cmimc_2025.yaml",
    },
    {
        "task_name": "BRUMO2025",
        "class_name": "BRUMO2025Benchmark",
        "dataset_name": "MathArena/brumo_2025",
        "expected_rows": 30,
        "config_file": "matharena_brumo_2025.yaml",
    },
    {
        "task_name": "SMT2025",
        "class_name": "SMT2025Benchmark",
        "dataset_name": "MathArena/smt_2025",
        "expected_rows": 53,
        "config_file": "matharena_smt_2025.yaml",
    },
]


def _task_ids(spec):
    return spec["task_name"]


def _wrapper_path(spec):
    return REPO / "eval/chat_benchmarks" / spec["task_name"] / "eval_instruct.py"


def _class_assignments(class_node):
    assignments = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.literal_eval(node.value)
    return assignments


@pytest.fixture(scope="module")
def loaded_matharena_modules():
    pytest.importorskip("datasets")
    pytest.importorskip("lm_eval")
    pytest.importorskip("sympy")

    from eval.chat_benchmarks import matharena_final_answer_common as common
    from eval.chat_benchmarks.Apex2025.eval_instruct import Apex2025Benchmark
    from eval.chat_benchmarks.ApexShortlist.eval_instruct import ApexShortlistBenchmark
    from eval.chat_benchmarks.BRUMO2025.eval_instruct import BRUMO2025Benchmark
    from eval.chat_benchmarks.CMIMC2025.eval_instruct import CMIMC2025Benchmark
    from eval.chat_benchmarks.SMT2025.eval_instruct import SMT2025Benchmark
    from eval.task import TaskManager

    classes = {
        "Apex2025": Apex2025Benchmark,
        "ApexShortlist": ApexShortlistBenchmark,
        "CMIMC2025": CMIMC2025Benchmark,
        "BRUMO2025": BRUMO2025Benchmark,
        "SMT2025": SMT2025Benchmark,
    }
    return {"common": common, "TaskManager": TaskManager, "classes": classes}


@pytest.mark.parametrize("spec", TASK_SPECS, ids=_task_ids)
def test_wrapper_modules_are_thin_task_declarations(spec):
    tree = ast.parse(_wrapper_path(spec).read_text())

    assert not [node for node in tree.body if isinstance(node, ast.Import)]
    import_nodes = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert len(import_nodes) == 1
    assert import_nodes[0].module == "eval.chat_benchmarks.matharena_final_answer_common"
    assert [alias.name for alias in import_nodes[0].names] == ["MathArenaFinalAnswerBenchmark"]

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert [class_node.name for class_node in classes] == [spec["class_name"]]

    class_node = classes[0]
    assert [base.id for base in class_node.bases if isinstance(base, ast.Name)] == [
        "MathArenaFinalAnswerBenchmark"
    ]
    assignments = _class_assignments(class_node)
    assert assignments["DATASET_NAME"] == spec["dataset_name"]
    assert assignments["EXPECTED_NUM_ROWS"] == spec["expected_rows"]
    assert isinstance(assignments["BENCHMARK_DESCRIPTION"], str)
    assert assignments["BENCHMARK_DESCRIPTION"]


@pytest.mark.parametrize("spec", TASK_SPECS, ids=_task_ids)
def test_single_task_config_selects_exact_task_with_safe_max_tokens(spec):
    config_path = REPO / "configs/single_task" / spec["config_file"]
    config = yaml.safe_load(config_path.read_text())

    assert config == {
        "max_tokens": 32768,
        "tasks": [{"task_name": spec["task_name"], "batch_size": 1}],
    }


def test_matharena_2025_task_discovery_registers_usable_instances(loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    TaskManager = loaded_matharena_modules["TaskManager"]
    task_names = [spec["task_name"] for spec in TASK_SPECS]

    manager = TaskManager(task_list=task_names)

    assert set(manager.available_tasks) == set(task_names)
    for spec in TASK_SPECS:
        benchmark = manager.get_benchmark(spec["task_name"])
        assert isinstance(benchmark, common.MathArenaFinalAnswerBenchmark)
        assert benchmark.__class__.__name__ == spec["class_name"]
        assert benchmark.dataset_name == spec["dataset_name"]
        assert benchmark.EXPECTED_NUM_ROWS == spec["expected_rows"]


def test_load_questions_normalizes_and_debug_slices(monkeypatch, loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    classes = loaded_matharena_modules["classes"]
    rows = [
        {"problem_idx": 1, "problem": "Problem 1", "answer": 5, "source": "source-a"},
        {"problem_idx": 2, "problem": "Problem 2", "answer": "x+1", "source": "source-b"},
        {"problem_idx": 3, "problem": "Problem 3", "answer": "7", "source": "source-c"},
    ]
    calls = {}

    def fake_load_dataset(dataset_name, **kwargs):
        calls["dataset_name"] = dataset_name
        calls["kwargs"] = kwargs
        return rows

    monkeypatch.setattr(common, "load_dataset", fake_load_dataset)

    questions = classes["ApexShortlist"](debug=True).load_questions()

    assert calls == {"dataset_name": "MathArena/apex-shortlist", "kwargs": {"split": "train"}}
    assert len(questions) == 2
    assert questions[0]["id"] == "1"
    assert questions[0]["answer"] == "5"
    assert questions[0]["dataset_name"] == "MathArena/apex-shortlist"
    assert questions[0]["source"] == "source-a"


def test_load_questions_passes_revision(monkeypatch, loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    classes = loaded_matharena_modules["classes"]
    calls = {}

    def fake_load_dataset(dataset_name, **kwargs):
        calls["dataset_name"] = dataset_name
        calls["kwargs"] = kwargs
        return [{"problem_idx": 1, "problem": "Problem", "answer": "42", "source": "source"}]

    monkeypatch.setattr(common, "load_dataset", fake_load_dataset)

    questions = classes["Apex2025"](dataset_revision="abc123", debug=True).load_questions()

    assert calls == {"dataset_name": "MathArena/apex_2025", "kwargs": {"split": "train", "revision": "abc123"}}
    assert questions[0]["id"] == "1"


def test_load_questions_validates_expected_rows(monkeypatch, loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    classes = loaded_matharena_modules["classes"]
    monkeypatch.setattr(
        common,
        "load_dataset",
        lambda *args, **kwargs: [{"problem_idx": 1, "problem": "Problem", "answer": "42", "source": "source"}],
    )

    with pytest.raises(ValueError, match="MathArena/apex_2025 expected 12 rows, got 1"):
        classes["Apex2025"]().load_questions()


@pytest.mark.parametrize(
    "task_name,row,metadata_key,metadata_value",
    [
        (
            "ApexShortlist",
            {"problem_idx": 7, "problem": "Problem", "answer": "1", "source": "smt-2025-p1"},
            "source",
            "smt-2025-p1",
        ),
        (
            "SMT2025",
            {"problem_idx": 8, "problem": "Problem", "answer": "2", "problem_type": ["Geometry"]},
            "problem_type",
            ["Geometry"],
        ),
    ],
)
def test_metadata_preserves_source_and_problem_type(
    task_name,
    row,
    metadata_key,
    metadata_value,
    loaded_matharena_modules,
):
    classes = loaded_matharena_modules["classes"]
    bench = classes[task_name](debug=True)
    example = bench._normalize_example(row, 0)
    metadata = bench._metadata_for_example(example)

    assert metadata["problem_id"] == str(row["problem_idx"])
    assert metadata["expected_answer"] == str(row["answer"])
    assert metadata["dataset_name"] == bench.dataset_name
    assert metadata[metadata_key] == metadata_value


@pytest.mark.parametrize("row", [{"answer": "1"}, {"problem": "Problem"}])
def test_normalize_example_rejects_missing_required_fields(row, loaded_matharena_modules):
    classes = loaded_matharena_modules["classes"]

    with pytest.raises(KeyError):
        classes["Apex2025"]()._normalize_example(row, 0)


def test_unsupported_native_passk_guard_is_task_local(loaded_matharena_modules):
    classes = loaded_matharena_modules["classes"]
    bench = classes["Apex2025"](num_samples=2)

    with pytest.raises(ValueError, match="does not implement native pass@k"):
        bench._ensure_single_sample_mode()


def test_generate_responses_builds_instances_and_extracts_answers(
    monkeypatch,
    loaded_matharena_modules,
):
    common = loaded_matharena_modules["common"]
    classes = loaded_matharena_modules["classes"]
    bench = classes["SMT2025"](debug=True, max_tokens=99)
    bench.n_repeat = 2
    examples = [
        bench._normalize_example(
            {"problem_idx": 3, "problem": "Compute 3.", "answer": "3", "problem_type": ["Algebra"]},
            0,
        ),
    ]
    answer_by_repeat = {
        0: {"3": "3"},
        1: {"3": "0"},
    }

    class CapturingModel:
        rank = 0
        world_size = 1
        model = "fake"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages):
            return messages[-1]["content"]

        def generate_until(self, prompts):
            self.calls.append(list(prompts))
            outputs = []
            for instance in prompts:
                answer = answer_by_repeat[instance.repeat_idx][instance.metadata["problem_id"]]
                outputs.append("Solution " + r"\boxed{" + answer + "}")
            return outputs

    model = CapturingModel()
    monkeypatch.setattr(bench, "load_questions", lambda: examples)

    result = bench.generate_responses(model)

    assert result["examples"][0]["model_outputs"] == [
        r"Solution \boxed{3}",
        r"Solution \boxed{0}",
    ]
    gold_answer = common.parse_answer("3")[0]
    assert common.check_answers(result["examples"][0]["model_answers"][0], gold_answer)
    assert not common.check_answers(result["examples"][0]["model_answers"][1], gold_answer)
    assert result["examples"][0]["label"] == []

    assert len(model.calls) == 2
    first_instance = model.calls[0][0]
    second_instance = model.calls[1][0]
    assert first_instance.repeat_idx == 0
    assert second_instance.repeat_idx == 1
    assert first_instance.metadata == {
        "problem_id": "3",
        "expected_answer": "3",
        "dataset_name": "MathArena/smt_2025",
        "problem_type": ["Algebra"],
    }
    prompt, gen_kwargs = first_instance.args
    assert prompt == (
        "Problem: Compute 3.\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "Answer:"
    )
    assert gen_kwargs["do_sample"] is False
    assert gen_kwargs["max_new_tokens"] == 99
    assert gen_kwargs["temperature"] == 0.7


def test_evaluate_responses_aggregates_repeated_runs_with_matharena_checker(loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    classes = loaded_matharena_modules["classes"]
    bench = classes["Apex2025"]()
    bench.n_repeat = 2
    results = {
        "examples": [
            {
                "answer": "1",
                "model_answers": [
                    common.extract_answer(r"\boxed{1}", False, True, False)[0],
                    common.extract_answer(r"\boxed{0}", False, True, False)[0],
                ],
                "label": [],
            },
            {
                "answer": "2",
                "model_answers": [
                    common.extract_answer(r"\boxed{2}", False, True, False)[0],
                    common.extract_answer(r"\boxed{2}", False, True, False)[0],
                ],
                "label": [],
            },
        ]
    }

    out = bench.evaluate_responses(results)

    assert out["num_total"] == 2
    assert out["solved_avg"] == pytest.approx(1.5)
    assert out["accuracy_avg"] == pytest.approx(0.75)
    assert out["accuracy_std_err"] == pytest.approx(0.176776695)
    assert out["num_repeat"] == 2
    assert out["run_stats"] == [
        {"repetition": 1, "num_total": 2, "num_solved": 2, "accuracy": 1.0},
        {"repetition": 2, "num_total": 2, "num_solved": 1, "accuracy": 0.5},
    ]
    assert [example["label"] for example in out["examples"]] == [[True, False], [True, True]]


def test_matharena_parser_smoke(loaded_matharena_modules):
    common = loaded_matharena_modules["common"]
    gold, _ = common.parse_answer(r"\frac{9\sqrt{30}}{4}")
    pred = common.extract_answer(
        r"Thus the final answer is \boxed{\frac{9\sqrt{30}}{4}}.",
        False,
        True,
        False,
    )[0]
    assert common.check_answers(pred, gold)
