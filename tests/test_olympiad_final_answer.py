import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance

from eval.chat_benchmarks.AMOBenchParser.eval_instruct import (
    AMOBenchParserBenchmark,
    extract_amo_prediction,
)
from eval.chat_benchmarks.AMOBenchParser.solver import solve_many_with_timeout
from eval.chat_benchmarks.OlymMATHEasy.eval_instruct import OlymMATHEasyBenchmark
from eval.chat_benchmarks.RIMON.eval_instruct import RIMONBenchmark
from eval.chat_benchmarks.final_answer_math import (
    _format_for_math_verify,
    exact_string_grade,
    extract_last_boxed,
    olymmath_grade,
    olymmath_string_fallback,
)
from eval.task import TaskManager


ROOT = Path(__file__).parents[1]
DATASETS = {
    "rimo": {
        "path": ROOT / "eval/chat_benchmarks/RIMON/data/rimo_n.jsonl",
        "sha256": "71a4d791210a954f03479c1541edd29e855c224fed786d45e44e48f7789fc08d",
        "rows": 335,
        "revision": "a6fb235c1eeb9592f66db5ac5fbbd2d0435fdb5e",
        "source": "https://huggingface.co/datasets/ziye2chen/RIMO/blob/a6fb235c1eeb9592f66db5ac5fbbd2d0435fdb5e/RIMO-N.jsonl",
        "license": "Apache-2.0",
        "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    "olymmath_easy": {
        "path": ROOT / "eval/chat_benchmarks/OlymMATHEasy/data/olymmath_en_easy.jsonl",
        "sha256": "1d96903b3017abd80388005ca9ad68c8074540ad6f41ee7cd92ac79cbcf5bc0a",
        "rows": 100,
        "revision": "2c6532ea2cf929ac1c421532af5951553eaee727",
        "source": "https://huggingface.co/datasets/RUC-AIBOX/OlymMATH/blob/2c6532ea2cf929ac1c421532af5951553eaee727/data/OlymMATH-EN-EASY.jsonl",
        "license": "MIT",
        "license_sha256": "ef797ff8176e6b1e3452388dbdddbaf505f26f0caaa43ade4018ce8401bf6deb",
        "grader_revision": "dd3d3042cceef1f1b0fb508ab50281a7ce8b60bf",
        "grader_source": "https://github.com/RUCAIBox/OlymMATH/blob/dd3d3042cceef1f1b0fb508ab50281a7ce8b60bf/local_tester.py",
    },
    "olymmath_hard": {
        "path": ROOT / "eval/chat_benchmarks/OlymMATHHard/data/olymmath_en_hard.jsonl",
        "sha256": "f43b48f5569a5e6d3068e110f8ce6c5ec641ba4d6ba06d9a91a1c944526b7be4",
        "rows": 100,
        "revision": "2c6532ea2cf929ac1c421532af5951553eaee727",
        "source": "https://huggingface.co/datasets/RUC-AIBOX/OlymMATH/blob/2c6532ea2cf929ac1c421532af5951553eaee727/data/OlymMATH-EN-HARD.jsonl",
        "license": "MIT",
        "license_sha256": "ef797ff8176e6b1e3452388dbdddbaf505f26f0caaa43ade4018ce8401bf6deb",
        "grader_revision": "dd3d3042cceef1f1b0fb508ab50281a7ce8b60bf",
        "grader_source": "https://github.com/RUCAIBox/OlymMATH/blob/dd3d3042cceef1f1b0fb508ab50281a7ce8b60bf/local_tester.py",
    },
    "amo_parser": {
        "path": ROOT / "eval/chat_benchmarks/AMOBenchParser/data/amo_bench_parser.jsonl",
        "sha256": "58ebd76884e88a7301abd4d788cdb93ebc570995677cab68ad0985345a4a3ff5",
        "rows": 39,
        "revision": "2f422616c25d862984408fbbfaed63a961e8e025",
        "source": "https://huggingface.co/datasets/meituan-longcat/AMO-Bench/blob/2f422616c25d862984408fbbfaed63a961e8e025/data/test-00000-of-00001.parquet",
        "license": "MIT",
        "license_sha256": "369cbf0306571034caa662fb7e2c8bc17ad4df3acce57c1df9fba19daf4faa14",
        "grader_revision": "52e1f378e4bcb0c593e860be38f9251b1a192571",
        "grader_source": "https://github.com/meituan-longcat/AMO-Bench/blob/52e1f378e4bcb0c593e860be38f9251b1a192571/utils.py",
        "source_rows": 50,
        "source_sha256": "eb08f5239c0cc092e13552c31e505a7fdb8e8140aa8e227beedd62fc442cde58",
    },
}


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("name", DATASETS)
def test_vendored_data_manifest_and_license_are_pinned(name):
    expected = DATASETS[name]
    path = expected["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    assert len(_rows(path)) == expected["rows"]

    manifest = json.loads((path.parent / "source_manifest.json").read_text(encoding="utf-8"))
    hash_key = "normalized_sha256" if name == "amo_parser" else "sha256"
    rows_key = "normalized_rows" if name == "amo_parser" else "rows"
    assert manifest[hash_key] == expected["sha256"]
    assert manifest[rows_key] == expected["rows"]
    for field in ("revision", "source", "license"):
        assert manifest[field] == expected[field]
    if "grader_revision" in expected:
        assert manifest["grader_revision"] == expected["grader_revision"]
        assert manifest["grader_source"] == expected["grader_source"]
    if name == "amo_parser":
        assert manifest["source_rows"] == expected["source_rows"]
        assert manifest["source_sha256"] == expected["source_sha256"]

    license_path = path.parent / "SOURCE_LICENSE"
    assert hashlib.sha256(license_path.read_bytes()).hexdigest() == expected["license_sha256"]


def test_source_distributions_and_ids():
    rimo = _rows(DATASETS["rimo"]["path"])
    easy = _rows(DATASETS["olymmath_easy"]["path"])
    hard = _rows(DATASETS["olymmath_hard"]["path"])
    amo = _rows(DATASETS["amo_parser"]["path"])

    assert Counter(row["type"] for row in rimo) == {
        "combinatorics": 96,
        "algebra": 95,
        "number theory": 86,
        "geometry": 58,
    }
    assert sum(str(row["answer"]).strip() in {"0", "1"} for row in rimo) == 96
    assert Counter(row["subject"] for row in easy) == {
        "Geometry": 33,
        "Combinatorics": 29,
        "Algebra": 25,
        "Number Theory": 13,
    }
    assert Counter(row["subject"] for row in hard) == {
        "Algebra": 25,
        "Geometry": 25,
        "Combinatorics": 25,
        "Number Theory": 25,
    }
    assert Counter(row["answer_type"] for row in amo) == {"number": 34, "set": 3, "variable": 2}
    variable_rows = {row["question_id"]: row for row in amo if row["answer_type"] == "variable"}
    assert set(variable_rows) == {5, 37}
    assert variable_rows[5]["verification_cases"] == [f"n={value}" for value in range(1, 21)]
    assert variable_rows[37]["verification_cases"] == [
        f"a={value},b={value + 1},c={value + 2}" for value in range(2, 19)
    ]

    for rows, id_field in ((rimo, "problem_id"), (easy, "unique_id"), (hard, "unique_id"), (amo, "question_id")):
        ids = [str(row[id_field]) for row in rows]
        assert len(ids) == len(set(ids))


def test_boxed_extraction_matches_released_balanced_and_fallback_paths():
    text = r"first \boxed{1}, nested \boxed{\frac{2}{3}}, final \boxed{\{4,5\}}"
    assert extract_last_boxed(text) == r"\{4,5\}"
    assert extract_last_boxed(r"double-escaped \\boxed{answer}") == "answer"
    assert extract_last_boxed(r"broken \boxed{1") == ""


def test_exact_string_grading_only_strips_outer_math_delimiters():
    assert exact_string_grade("50", "$50$").correct
    assert exact_string_grade(r"2^{2024}-1", r"2^{2024}-1").correct
    assert not exact_string_grade("1/2", r"\frac{1}{2}").correct
    assert exact_string_grade("50", "").method == "missing_answer"


def test_olymmath_string_fallback_matches_released_behavior():
    assert olymmath_string_fallback("13", "3")
    assert olymmath_string_fallback(r"\frac12", "12")
    assert not olymmath_string_fallback("13", "7")


def test_olymmath_prompt_formatting_and_sampling_match_source():
    benchmark = OlymMATHEasyBenchmark(debug=True)
    example = benchmark.load_questions()[0]
    expected_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + example["problem"]
    assert benchmark.prompt_for(example) == expected_prompt

    instance = benchmark._build_instances(_RecordingAnswerLM(), [example], True, benchmark.seed, 0)[0]
    assert instance.args[0] == expected_prompt
    assert instance.args[1]["temperature"] == 0.6
    assert instance.args[1]["top_p"] == 0.95

    assert _format_for_math_verify("$x$") == "$x$"
    assert _format_for_math_verify("$$x$$") == "$$x$$"
    assert _format_for_math_verify(r"\(x\)") == r"$\(x\)$"


def test_amo_prediction_extraction_matches_source_markers():
    output = "reasoning</think>\n### The final answer is: $\\boxed{6}$\n---\nextra"
    assert extract_amo_prediction(output, "number") == "$\\boxed{6}$ --- extra"
    assert extract_amo_prediction(r"### Final answer: \left\{1,2\right\}", "set") == r"\{1,2\}"


def test_all_released_olymmath_answers_pass_with_math_verify():
    for key in ("olymmath_easy", "olymmath_hard"):
        for row in _rows(DATASETS[key]["path"]):
            grade = olymmath_grade(row["answer"], row["answer"])
            assert grade.correct, row["unique_id"]
            assert grade.method == "math_verify", row["unique_id"]


def test_all_released_amo_parser_answers_pass():
    amo = AMOBenchParserBenchmark()
    rows = amo.load_questions()
    for row in rows:
        prediction = extract_amo_prediction(row["answer"], row["answer_type"])
        assert amo.grade_answer(row["answer"], prediction, row).correct, row["question_id"]

    representative_rows = {
        answer_type: next(row for row in rows if row["answer_type"] == answer_type)
        for answer_type in {"number", "set", "variable"}
    }
    for row in representative_rows.values():
        wrong = extract_amo_prediction("definitely-wrong", row["answer_type"])
        assert not amo.grade_answer(row["answer"], wrong, row).correct, row["question_id"]


def test_olymmath_grader_uses_math_verify_from_eval_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        grade = executor.submit(olymmath_grade, r"\frac{1}{2}", "0.5").result(timeout=10)
        wrong = executor.submit(olymmath_grade, r"\frac{1}{2}", "definitely-wrong").result(timeout=10)
    assert grade.correct
    assert grade.method == "math_verify"
    assert not wrong.correct


def test_tasks_are_discoverable_without_api_key_or_math_extra(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "math_verify", None)
    names = ["RIMON", "OlymMATHEasy", "OlymMATHHard", "AMOBenchParser"]
    manager = TaskManager(task_list=names)
    assert set(manager.available_tasks) == set(names)
    assert not any(manager.requires_annotator_model(name) for name in names)
    with pytest.raises(RuntimeError, match="pip install -e"):
        olymmath_grade("1", "1")


def test_task_manager_loaded_amo_variable_grader_works_from_eval_thread():
    manager = TaskManager(task_list=["AMOBenchParser"])
    benchmark = manager.get_benchmark("AMOBenchParser")
    example = next(row for row in benchmark.load_questions() if row["answer_type"] == "variable")
    prediction = benchmark.extract_answer(example["answer"], example)
    with ThreadPoolExecutor(max_workers=1) as executor:
        grade = executor.submit(benchmark.grade_answer, example["answer"], prediction, example).result(timeout=30)
    assert grade.correct


def test_variable_solver_timeout_terminates_process():
    from sympy import Symbol

    existing_children = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(TimeoutError):
        solve_many_with_timeout([Symbol("x")], timeout_seconds=0)
    assert {process.pid for process in multiprocessing.active_children()} <= existing_children


class _RecordingAnswerLM:
    rank = 0
    world_size = 1

    def __init__(self, only_first_sample_correct=False):
        self.only_first_sample_correct = only_first_sample_correct
        self.calls = []

    def apply_chat_template(self, messages):
        return messages[-1]["content"]

    def generate_until(self, instances):
        self.calls.append(
            [
                {
                    "prompt": instance.args[0],
                    "generation_kwargs": dict(instance.args[1]),
                    "problem_id": instance.doc.get("problem_id"),
                    "repeat_idx": getattr(instance, "repeat_idx", 0),
                }
                for instance in instances
            ]
        )
        outputs = []
        for instance in instances:
            repeat_idx = getattr(instance, "repeat_idx", 0)
            answer = str(instance.doc["answer"])
            if self.only_first_sample_correct and repeat_idx > 0:
                answer = "definitely-wrong"
            outputs.append(f"work \\boxed{{{answer}}}")
        return outputs


def test_rimo_single_sample_generation_metrics_and_serialization():
    benchmark = RIMONBenchmark(debug=True)
    model = _RecordingAnswerLM()
    result = benchmark.evaluate_responses(benchmark.generate_responses(model))
    assert result["num_total"] == 2
    assert result["num_solved"] == 2
    assert result["accuracy"] == 1.0
    assert result["subgroups"]["type"] == {"algebra": {"num_total": 2, "num_solved": 2, "accuracy": 1.0}}
    assert result["subgroups"]["answer_class"] == {
        "binary": {"num_total": 1, "num_solved": 1, "accuracy": 1.0},
        "non_binary": {"num_total": 1, "num_solved": 1, "accuracy": 1.0},
    }
    assert all(example["grade"]["method"] == "exact_string" for example in result["examples"])
    assert len(model.calls) == 1
    assert [request["problem_id"] for request in model.calls[0]] == ["2023a1", "2023a2"]
    assert all(request["generation_kwargs"]["do_sample"] is False for request in model.calls[0])

    samples = benchmark.to_samples(result)
    assert len(samples) == 2
    assert samples[0]["arguments"][0][0] == benchmark.prompt_for(result["examples"][0])
    assert samples[0]["target"] == result["examples"][0]["answer"]
    assert samples[0]["filtered_resps"] == [result["examples"][0]["answer"]]
    assert all(len(samples[0][key]) == 64 for key in ("doc_hash", "prompt_hash", "target_hash"))
    json.dumps(samples)


def test_rimo_native_pass_at_k_generation_and_grades():
    benchmark = RIMONBenchmark(debug=True, num_samples=3, pass_at_k=[1, 2])
    model = _RecordingAnswerLM(only_first_sample_correct=True)
    result = benchmark.evaluate_responses(benchmark.generate_responses(model))
    assert result["num_correct"] == [1, 1]
    assert result["pass@1"] == pytest.approx(1 / 3)
    assert result["pass@2"] == pytest.approx(2 / 3)
    assert all(len(example["sample_grades"]) == 3 for example in result["examples"])
    assert all(
        [grade["correct"] for grade in example["sample_grades"]] == [True, False, False]
        for example in result["examples"]
    )
    assert len(model.calls) == 3
    assert [call[0]["repeat_idx"] for call in model.calls] == [0, 1, 2]
    assert all(request["generation_kwargs"]["do_sample"] is True for call in model.calls for request in call)
    assert all(request["generation_kwargs"]["top_p"] == 1.0 for call in model.calls for request in call)


def test_lazy_lm_eval_backend_detection_normalizes_generation_kwargs():
    benchmark = RIMONBenchmark(debug=True)

    openai_type = type(
        "OpenAIChatCompletion",
        (),
        {"__module__": "lm_eval.models.openai_completions", "model": "gpt-4o"},
    )
    openai_instance = Instance(
        "generate_until",
        {},
        ("prompt", {"seed": [1, 2, 3, 4], "max_new_tokens": 20000}),
        0,
    )
    benchmark._normalize_model_args(openai_type(), [openai_instance])
    assert openai_instance.args[1] == {"seed": 1, "max_tokens": 16384}

    vllm_type = type("VLLM", (), {"__module__": "lm_eval.models.vllm_causallms"})
    vllm_instance = Instance(
        "generate_until",
        {},
        ("prompt", {"seed": [1, 2, 3, 4], "max_new_tokens": 99}),
        0,
    )
    benchmark._normalize_model_args(vllm_type(), [vllm_instance])
    assert vllm_instance.args[1] == {"seed": 1, "max_gen_toks": 99}
