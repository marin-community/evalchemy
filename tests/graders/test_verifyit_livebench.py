"""LiveBench normalization, strict instruction scores and incomplete batch failures."""

import json
import subprocess

import pytest
from verifyit.grade import InvalidTask

from eval.graders.verifyit_livebench import JudgmentBatch, grade_cta, grade_retained


def question(**changes):
    row = {
        "question_id": "q1",
        "category": "data_analysis",
        "task": "cta",
        "ground_truth": "City",
        "turns": ["Classify the column"],
        "livebench_release_date": "2024-06-24",
        "livebench_removal_date": "",
    }
    row.update(changes)
    return row


def response(question_id="q1", text="City", model="fixture"):
    return {"question_id": question_id, "model_id": model, "choices": [{"index": 0, "turns": [text]}]}


def test_cta_normalized_suffix_and_case_are_preserved():
    assert grade_cta(question(), "The type is CITY.").reward == 1
    assert grade_cta(question(), r"\boxed{\text{City}}").reward == 1
    assert grade_cta(question(), "City then Number").reward == 0
    with pytest.raises(InvalidTask):
        grade_cta(question(ground_truth="!!!"), "")


def test_incomplete_later_task_removes_earlier_positive_judgment(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    second.write_text('{"score": 1}\n')
    with pytest.raises(InvalidTask):
        with JudgmentBatch([response(), response("q2")]) as batch:
            batch(questions=[question()], output_file=first, model_list=["fixture"])
            assert json.loads(first.read_text())["score"] == 1
            batch(
                questions=[question(question_id="q2", ground_truth="!!!")], output_file=second, model_list=["fixture"]
            )
    assert not first.exists()
    assert not second.exists()


def test_mixed_model_responses_cannot_be_silently_ignored():
    with pytest.raises(InvalidTask):
        JudgmentBatch([response(), response("q2", model="other")])


@pytest.mark.parametrize(("text", "expected"), [("alpha beta", 1), ("alpha", 0.25), ("gamma", 0)])
def test_instruction_prompt_and_fractional_scores_use_actual_predicates(text, expected):
    row = question(
        category="instruction_following",
        task="summarize",
        ground_truth=None,
        instruction_id_list=["keywords:existence", "keywords:existence"],
        kwargs=[{"keywords": ["alpha"]}, {"keywords": ["beta"]}],
    )
    verdict = grade_retained(row, text)
    assert verdict.reward == expected
    assert len(verdict.detail["instruction_flags"]) == 2
    assert all(item["verdict"]["status"] == "scored" for item in verdict.detail["instruction_verdicts"])


@pytest.mark.parametrize(
    "identifier,kwargs,text,expected",
    [
        ("language:response_language", {"language": "en"}, "12345", 0),
        ("change_case:english_capital", {}, "THIS IS A COMPLETE ENGLISH SENTENCE ABOUT A BEAUTIFUL GARDEN.", 1),
        ("change_case:english_capital", {}, "This is a complete English sentence about a beautiful garden.", 0),
        ("change_case:english_lowercase", {}, "this is a complete english sentence about a beautiful garden.", 1),
    ],
)
def test_language_observations_require_detectable_language_and_case(identifier, kwargs, text, expected):
    row = question(
        category="instruction_following",
        task="summarize",
        ground_truth=None,
        instruction_id_list=[identifier],
        kwargs=[kwargs],
    )
    assert grade_retained(row, text).reward == expected


def test_coding_without_trusted_tests_is_invalid_task():
    row = question(
        category="coding",
        task="LCB_generation",
        ground_truth=None,
        public_test_cases="[]",
        private_test_cases="[]",
        original_json={"metadata": "{}"},
    )
    with pytest.raises(InvalidTask, match="nonempty trusted test"):
        grade_retained(row, "```python\nprint(1)\n```")
    row.update(
        public_test_cases='[{"input":"0","output":"1","testtype":"functional"}]',
        private_test_cases='[{"input":"0","output":"NaN","testtype":"functional"}]',
        original_json={"metadata": '{"func_name":"answer"}'},
    )
    with pytest.raises(InvalidTask, match="nonfinite"):
        grade_retained(row, "")
    row.update(private_test_cases="[]", partial_solution=0)
    with pytest.raises(InvalidTask, match="partial solution"):
        grade_retained(row, "```python\nclass Solution:\n def answer(self, value): return 1\n```")


def test_hf_empty_release_population_cannot_reuse_stale_positive_file(tmp_path, monkeypatch):
    from datasets import Dataset
    from eval.chat_benchmarks.LiveBench import eval_instruct as module

    rows = [question(), question(question_id="q2")]
    dataset = Dataset.from_list(rows)
    monkeypatch.setattr(
        module,
        "get_categories_tasks",
        lambda _: ({"data_analysis": dataset}, {"data_analysis": ["cta", "empty_task"]}),
    )
    stale = tmp_path / "live_bench/data_analysis/empty_task/model_judgment/ground_truth_judgment.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"question_id": "q1", "task": "cta", "category": "data_analysis", "score": 1}) + "\n")
    benchmark = module.LiveBenchBenchmark(dataset_name="live_bench/data_analysis", verifyit_enabled=True)
    benchmark.data_path = str(tmp_path)
    result = benchmark.evaluate_responses([response(), response("q2", "Number")])
    assert result["num_questions"] == 2
    assert result["metrics"]["global_average"] == 50
    assert not stale.exists()


@pytest.mark.parametrize(("task", "gold"), [("imo", "1,2"), ("connections", "alpha,beta,gamma,delta")])
def test_incomplete_candidate_box_is_scored_zero(task, gold):
    assert grade_retained(question(task=task, ground_truth=gold), r"\boxed{").reward == 0


@pytest.mark.parametrize(
    ("task", "gold"),
    [
        ("imo", "not a sequence"),
        ("plot_unscrambling", "..."),
        ("connections", ",,"),
        ("tablejoin", "{}"),
        ("amps_hard", r"\left\right"),
    ],
)
def test_malformed_trusted_reference_cannot_be_scored_as_candidate_failure(task, gold):
    with pytest.raises(InvalidTask):
        grade_retained(question(task=task, ground_truth=gold), "")


def test_empty_amps_answer_is_scored_zero():
    assert grade_retained(question(task="amps_hard", ground_truth="42"), "").reward == 0


@pytest.mark.parametrize("failure", [ImportError("missing HTML parser"), RuntimeError("reader runtime failed")])
def test_table_reader_dependency_failure_is_not_an_invalid_task(monkeypatch, failure):
    from eval.graders.verifyit_livebench import retained_score
    from livebench.process_results.data_analysis.tablereformat import utils

    def unavailable_reader(*args, **kwargs):
        raise failure

    monkeypatch.setattr(utils.pd, "read_html", unavailable_reader)
    row = question(
        task="tablereformat",
        ground_truth="<table><tr><th>x</th></tr><tr><td>1</td></tr></table>",
        turns=["Please convert the Input Table from csv format to html format"],
    )
    with pytest.raises(type(failure), match=str(failure)):
        retained_score(row, row["ground_truth"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [("AAAA", 1), (r"\boxed{a}", 1), ("The answer is 42", 1), ("The answer is 17", 0), (r"\boxed{ a }", 0)],
)
def test_contest_answer_forms_preserve_literal_boundaries(text, expected):
    row = question(category="math", task="amc", ground_truth="A", turns=[r"$\textbf{(A)}42\qquad\textbf{(B)}17$"])
    assert grade_retained(row, text).reward == expected


def test_connections_repeated_group_cannot_replace_missing_group():
    row = question(task="connections", ground_truth="a,b,c,d,e,f,g,h", livebench_release_date="2024-11-25")
    assert grade_retained(row, "<solution>a,b,c,d,e,f,g,h</solution>").reward == 1
    assert grade_retained(row, "<solution>a,b,c,d,a,b,c,d</solution>").reward == 0.5
    assert grade_retained(row, "<solution>a,b,c,d</solution>").reward == 0.5
    assert grade_retained(row, "<solution>w,x,y,z</solution>").reward == 0


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("class Solution:\n def answer(self, value): return value + 1", 1),
        ("class Solution:\n def answer(self, value): return True", 0),
        ("class Solution:\n def answer(self, value): raise RuntimeError('failed')", 0),
        ("invalid python syntax !", 0),
        ("class Solution:\n def answer(self, value):\n  while True: pass", 0),
    ],
)
def test_coding_core_compares_function_results_and_rejects_candidate_failure(completion, expected):
    row = question(
        category="coding",
        task="LCB_generation",
        public_test_cases=json.dumps([{"input": "0", "output": "1", "testtype": "functional"}]),
        private_test_cases="[]",
        original_json={"metadata": '{"func_name": "answer"}'},
    )
    verdict = grade_retained(row, "```python\n" + completion + "\n```")
    assert verdict.reward == expected
    container = verdict.detail["container_name"]
    inspection = subprocess.run(["docker", "inspect", container], capture_output=True, timeout=10)
    assert inspection.returncode != 0


@pytest.mark.parametrize(
    ("task", "reference", "release", "text", "reward"),
    [
        ("zebra_puzzle", "three", "2024-06-24", "***3***", 1),
        ("zebra_puzzle", "red, 2", "2024-11-25", "<solution>red, position 3</solution>", 0.25),
        ("zebra_puzzle", "red, 2", "2024-11-25", "<solution>red, position 2</solution>", 1),
        ("web_of_lies_v2", "yes, no, yes", "2024-06-24", "**yes** **no** **yes**", 1),
        ("web_of_lies_v2", "yes, no, yes", "2024-06-24", r"\boxed{yes, no, yes, no}", 0),
        ("spatial", "triangle", "2024-07-26", "**equilateral triangle**", 1),
        ("spatial", "triangle", "2024-07-26", "**circle square triangle**", 0),
    ],
)
def test_reasoning_answer_formats_and_partial_credit_use_core(task, reference, release, text, reward):
    row = question(category="reasoning", task=task, ground_truth=reference, livebench_release_date=release)
    assert grade_retained(row, text).reward == reward


def test_zebra_empty_trusted_member_cannot_earn_partial_credit():
    row = question(category="reasoning", task="zebra_puzzle", ground_truth="red,", livebench_release_date="2024-11-25")
    with pytest.raises(InvalidTask, match="nonempty answers"):
        grade_retained(row, "<solution>red, anything</solution>")


@pytest.mark.parametrize(
    ("text", "reward"),
    [
        ('b,a\n,1.0000005\n word ,2\n', 1),
        ('a,b\n1.0000011,\n2,word\n', 0),
        ('a,b\n2,word\n1,\n', 0),
        ('a,b\n,\n2,word\n', 0),
        ('a,b,c\n1,,extra\n2,word,extra\n', 0),
    ],
)
def test_table_core_preserves_order_nulls_and_numeric_tolerance(text, reward):
    row = question(
        task="tablereformat",
        ground_truth="a,b\n1,\n2,word\n",
        turns=["Please convert the Input Table from json format to csv format"],
    )
    assert grade_retained(row, text).reward == reward


def test_tsv_successful_wrong_parse_does_not_retry_embedded_table():
    from livebench.process_results.data_analysis.tablereformat import utils

    prompt = "Please convert the Input Table from json format to tsv format"
    reference = "a\tb\n1\t2\n"
    candidate = "Explanation\na\tb\n1\t2\n"
    assert utils.read_df_func("tsv", candidate) is not None
    assert utils.table_process_results(prompt, reference, candidate) == 0
    assert grade_retained(question(task="tablereformat", ground_truth=reference, turns=[prompt]), candidate).reward == 0

@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("{'a': 'x', 'b': 'y'}", 1.0),
        ("{'a': 'x'}", 0.67),
        ("{'a': 'x', 'b': 'wrong'}", 0.5),
        ("{'a': 'x', 'b': None}", 0.67),
        ("{'a': 'x', 'b': ''}", 0.0),
    ],
)
def test_table_join_core_overlap_and_conservative_malformed_labels(candidate, expected):
    from eval.graders.verifyit_livebench_tables import grade_join
    from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results

    row = question(task="tablejoin", ground_truth="{'a': 'x', 'b': 'y'}")
    assert grade_join(row, candidate).reward == expected
    if "''" not in candidate:
        assert joinmap_process_results(None, row["ground_truth"], candidate) == expected


def test_table_join_invalid_reference_precedes_unparseable_candidate():
    from eval.graders.verifyit_livebench_tables import grade_join

    with pytest.raises(InvalidTask):
        grade_join(question(task="tablejoin", ground_truth="{'a': None}"), "not a mapping")
