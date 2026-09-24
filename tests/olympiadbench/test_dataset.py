import math
from types import SimpleNamespace

import datasets
import pytest

from eval.graders import answer_equivalence
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.chat_benchmarks.OlympiadBenchFull.eval_instruct import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_SPLIT,
    OlympiadBenchFullBenchmark,
)
from eval.chat_benchmarks.OlympiadBenchDeterministic.eval_instruct import OlympiadBenchDeterministicBenchmark
from eval.task import TaskManager
from eval.graders.answer_extraction import EmptyResponseError, MissingAnswerError


class _IncorrectJudgeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="incorrect"))])


@pytest.fixture(autouse=True)
def _judge_credentials(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "olympiadbench-test-key")
    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", _IncorrectJudgeClient)


def test_olympiadbench_aliases_expose_the_legacy_subset_and_full_text_only_set():
    manager = TaskManager(task_list=["OlympiadBench", "OlympiadBenchFull"])

    assert set(manager.tasks) == {"OlympiadBench", "OlympiadBenchFull"}
    assert manager.get_benchmark("OlympiadBench").n_repeat == 10
    assert manager.get_benchmark("OlympiadBenchFull").dataset_revision == DEFAULT_DATASET_REVISION


def test_olympiadbench_uses_judge_credentials_without_candidate_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    manager = TaskManager(task_list=["OlympiadBench", "OlympiadBenchFull"])

    assert manager.load_failures == {}
    assert manager.requires_judge_credentials("OlympiadBench")
    assert manager.requires_judge_credentials("OlympiadBenchFull")


def test_deterministic_olympiadbench_scores_without_judge_credentials(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = TaskManager(task_list=["OlympiadBenchDeterministic"])

    assert manager.load_failures == {}
    assert not manager.requires_judge_credentials("OlympiadBenchDeterministic")
    assert manager.get_benchmark("OlympiadBenchDeterministic").n_repeat == 1
    benchmark = OlympiadBenchDeterministicBenchmark(n_repeat=1)
    results = benchmark.evaluate_responses(
        {
            "examples": [
                {"problem": "Compute one half.", "answer": ["0.5"], "model_answer": r"\frac{1}{2}"},
                {"problem": "Convert the length.", "answer": ["100 cm"], "model_answer": "1 m"},
            ]
        }
    )

    assert results["accuracy"] == 0.5
    assert results["num_judged_by_llm"] == 0
    assert results["judge_model"] is None


def test_olympiadbench_aliases_share_explicit_judge_model():
    manager = TaskManager(
        task_list=["OlympiadBench", "OlympiadBenchFull"],
        annotator_model="judge-model",
    )

    assert manager.get_benchmark("OlympiadBench").judge_config.model == "judge-model"
    assert manager.get_benchmark("OlympiadBenchFull").judge_config.model == "judge-model"


def test_legacy_olympiadbench_subset_reports_aime_style_repeat_standard_error():
    benchmark = OlympiadBenchBenchmark(n_repeat=2)
    results = benchmark.evaluate_responses(
        {
            "examples": [
                {"answer": ["1"], "model_answers": ["1", "0"]},
                {"answer": ["2"], "model_answers": ["2", "0"]},
            ]
        }
    )

    assert results["num_total"] == 2
    assert results["num_repeat"] == 2
    assert results["accuracy_avg"] == 0.5
    assert results["accuracy_std_err"] == 0.5 / math.sqrt(2)


def test_full_olympiadbench_loads_the_pinned_text_only_dataset(monkeypatch):
    rows = [
        {
            "question_id": "text-only",
            "subfield": "Algebra",
            "context": "Use the lemma.",
            "question": "Find x.",
            "final_answer": ["1"],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": "Numerical",
            "error": None,
            "source": "OE_TO_maths_en_COMP",
        },
        {
            "question_id": "multimodal",
            "subfield": "Mechanics",
            "context": None,
            "question": "Read the diagram.",
            "final_answer": ["2"],
            "is_multiple_answer": False,
            "unit": None,
            "answer_type": "Numerical",
            "error": None,
            "source": "OE_MM_physics_en_COMP",
        },
    ]
    request = {}

    def load_dataset(name, *, split, revision, cache_dir=None):
        request.update(name=name, split=split, revision=revision, cache_dir=cache_dir)
        return rows

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    questions = OlympiadBenchFullBenchmark().load_questions()

    assert request == {
        "name": DEFAULT_DATASET,
        "split": DEFAULT_SPLIT,
        "revision": DEFAULT_DATASET_REVISION,
        "cache_dir": None,
    }
    assert [question["id"] for question in questions] == ["text-only"]


def test_full_olympiadbench_reports_dataset_provenance_and_sample_standard_error():
    benchmark = OlympiadBenchFullBenchmark()
    results = benchmark.evaluate_responses(
        {
            "examples": [
                {"answer": ["1"], "model_answer": "1"},
                {"answer": ["2"], "model_answer": "0"},
                {"answer": ["3"], "model_answer": "0"},
            ]
        }
    )

    assert results["dataset_name"] == DEFAULT_DATASET
    assert results["dataset_revision"] == DEFAULT_DATASET_REVISION
    assert results["dataset_split"] == DEFAULT_SPLIT
    assert results["dataset_num_samples"] == 3
    assert results["accuracy"] == 1 / 3
    assert results["accuracy_stderr"] == pytest.approx(1 / 3)


def test_olympiadbench_accepts_sympy_equivalence_without_contacting_judge(monkeypatch):
    class UnexpectedJudgeClient:
        def __init__(self, **kwargs):
            raise AssertionError("symbolically equivalent answers must not call the LLM judge")

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", UnexpectedJudgeClient)
    benchmark = OlympiadBenchBenchmark(n_repeat=1)

    results = benchmark.evaluate_responses(
        {"examples": [{"problem": "Compute one half.", "answer": ["0.5"], "model_answer": r"\frac{1}{2}"}]}
    )

    assert results["accuracy"] == 1.0
    assert results["num_judged_by_llm"] == 0


@pytest.mark.parametrize("benchmark_class", [OlympiadBenchBenchmark, OlympiadBenchFullBenchmark])
def test_olympiadbench_uses_llm_fallback_for_unit_equivalence(monkeypatch, benchmark_class):
    class UnitJudgeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            assert "100 cm" in prompt
            assert "1 m" in prompt
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))])

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", UnitJudgeClient)
    benchmark = benchmark_class(n_repeat=1) if benchmark_class is OlympiadBenchBenchmark else benchmark_class()

    results = benchmark.evaluate_responses(
        {"examples": [{"problem": "Convert the length.", "answer": ["100 cm"], "model_answer": "1 m"}]}
    )

    assert results["accuracy"] == 1.0
    assert results["num_judged_by_llm"] == 1
    assert results["examples"][0]["judge_label"] == "correct"


def test_olympiadbench_pass_at_k_uses_hybrid_grader_for_every_completion(monkeypatch):
    class CorrectJudgeClient(_IncorrectJudgeClient):
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))])

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", CorrectJudgeClient)
    benchmark = OlympiadBenchBenchmark(num_samples=2, pass_at_k=[1, 2], n_repeat=1)

    results = benchmark.evaluate_responses(
        {
            "pass_at_k": True,
            "examples": [
                {
                    "problem": "Convert the length.",
                    "answer": ["100 cm"],
                    "model_answers": ["100", "1 m"],
                }
            ],
        }
    )

    assert results["pass@1"] == 1.0
    assert results["pass@2"] == 1.0
    assert results["num_graded_by_minerva"] == 1
    assert results["num_judged_by_llm"] == 1


def test_olympiadbench_records_llm_judge_failure_without_accepting_answer(monkeypatch):
    class FailingJudgeClient(_IncorrectJudgeClient):
        async def create(self, **kwargs):
            raise TimeoutError("judge unavailable")

    monkeypatch.setattr(answer_equivalence, "AsyncOpenAI", FailingJudgeClient)
    benchmark = OlympiadBenchBenchmark(n_repeat=1)

    results = benchmark.evaluate_responses(
        {"examples": [{"problem": "Find x.", "answer": ["2"], "model_answer": "3"}]}
    )

    assert results["accuracy"] == 0.0
    assert results["num_judge_failed"] == 1
    assert results["examples"][0]["judge_error"]["exception_type"] == "TimeoutError"


def test_olympiadbench_does_not_extract_boxed_answer_from_repeated_question():
    output = r"Solution. \boxed{17}" + "\nA.2 Solve the next part. " + r"Solution. \boxed{42}"

    assert OlympiadBenchBenchmark().extract_answer(output) == "17"


def test_olympiadbench_reports_unboxed_answer_as_typed_extraction_error():
    with pytest.raises(MissingAnswerError):
        OlympiadBenchBenchmark().extract_answer("The final answer is 17.")


def test_olympiadbench_distinguishes_empty_response_from_missing_answer_syntax():
    with pytest.raises(EmptyResponseError):
        OlympiadBenchBenchmark().extract_answer("")


def test_olympiadbench_records_unboxed_answer_error_and_scores_it_incorrect():
    class Model:
        rank = 0
        world_size = 1

        @staticmethod
        def apply_chat_template(messages):
            return messages

        @staticmethod
        def generate_until(requests):
            assert requests[0].args[1]["until"]
            return ["The final answer is 17."]

    benchmark = OlympiadBenchBenchmark(n_repeat=1)
    benchmark.load_questions = lambda: [{"problem": "Find x.", "answer": ["17"]}]

    generated = benchmark.generate_responses(Model())
    scored = benchmark.evaluate_responses(generated)
    sample = benchmark.to_samples(generated, scored)[0]

    assert generated["examples"][0]["answer_extraction_error"]["type"] == "MissingAnswerError"
    assert scored["accuracy"] == 0.0
    assert sample["answer_extraction_errors"][0]["type"] == "MissingAnswerError"
    assert "answer_extraction_error" not in sample["doc"]
