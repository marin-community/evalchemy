"""Exercise the native harness boundary through its real response filters."""

import math
from pathlib import Path
from types import FunctionType

import pytest
from datasets import Dataset, DatasetDict
from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate, simple_evaluate
from lm_eval.tasks._yaml_loader import load_yaml
from verifyit.grade import InvalidTask

from eval.lm_eval_compat import setup_parser
from eval.resume.lm_eval_native import resume_simple_evaluate


class Responses(LM):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses

    def generate_until(self, requests, **kwargs):
        return [self.responses[request.doc_id] for request in requests]

    def loglikelihood(self, requests, **kwargs):
        raise AssertionError("Unexpected likelihood request")

    def loglikelihood_rolling(self, requests, **kwargs):
        raise AssertionError("Unexpected rolling request")


def short_answer_task(name, references):
    filename = "nq_open.yaml" if name == "nq_open" else "default.yaml"
    path = Path(__file__).parents[2] / "eval" / "lm_eval_tasks" / name / filename
    config = load_yaml(path, resolve_func=True)
    docs = [
        {"question": "What is the answer?", "answer": refs if name == "nq_open" else {"aliases": refs}}
        for refs in references
    ]
    config.update(
        num_fewshot=0,
        training_split=None,
        validation_split=None,
        fewshot_split=None,
        test_split="test",
        custom_dataset=lambda **kwargs: DatasetDict(test=Dataset.from_list(docs)),
    )
    return ConfigurableTask(config=config)


@pytest.mark.parametrize("name", ["nq_open", "triviaqa"])
def test_reserved_invalid_extraction_cannot_match_a_real_reference(name):
    outputs = [None, [], "", "Answer: invalid"]
    for enabled in [False, True]:
        result = evaluate(
            Responses(outputs),
            {"case": short_answer_task(name, [["invalid"]] * len(outputs))},
            bootstrap_iters=0,
            log_samples=True,
            verifyit_enabled=enabled,
        )
        for filter_name in ["strict_answer", "extract_answer"]:
            scores = [sample["exact_match"] for sample in result["samples"]["case"] if sample["filter"] == filter_name]
            assert scores == ([0, 0, 0, 1] if enabled else [1, 1, 1, 1])
        assert result.get("config", {}).get("verifyit_enabled", False) is enabled


@pytest.mark.parametrize("name", ["nq_open", "triviaqa"])
@pytest.mark.parametrize("references", [None, []])
def test_invalid_trusted_aliases_are_not_hidden_by_missing_answers(name, references):
    with pytest.raises(InvalidTask, match="reference aliases"):
        evaluate(
            Responses([None]),
            {"case": short_answer_task(name, [references])},
            bootstrap_iters=0,
            verifyit_enabled=True,
        )


def test_harness_cli_opt_in_reaches_native_simple_evaluate():
    args = setup_parser().parse_args(["--verifyit_harness"])
    result = resume_simple_evaluate(
        simple_evaluate,
        model=Responses([None]),
        tasks=[short_answer_task("nq_open", [["invalid"]])],
        bootstrap_iters=0,
        log_samples=True,
        verifyit_enabled=args.verifyit_harness,
    )
    assert result["config"]["verifyit_enabled"] is True
    assert result["results"]["nq_open"]["exact_match,strict_answer"] == 0
    assert result["results"]["nq_open"]["exact_match,extract_answer"] == 0


def test_global_opt_in_does_not_fall_back_to_a_custom_source_scorer():
    path = Path(__file__).parents[2] / "eval" / "lm_eval_tasks" / "gsm8k" / "gsm8k.yaml"
    config = load_yaml(path, resolve_func=True)
    config.update(
        num_fewshot=0,
        training_split=None,
        fewshot_split=None,
        custom_dataset=lambda **kwargs: DatasetDict(
            test=Dataset.from_list([{"question": "What is two plus two?", "answer": "Two plus two is four. #### 4"}])
        ),
    )
    with pytest.raises(InvalidTask, match="No native verifyit contract"):
        evaluate(
            Responses(["#### 4"]),
            {"case": ConfigurableTask(config=config)},
            bootstrap_iters=0,
            verifyit_enabled=True,
        )


@pytest.mark.parametrize("name", ["nq_open", "triviaqa"])
def test_missing_and_punctuation_only_answers_cannot_match_normalized_empty_alias(name):
    for enabled in [False, True]:
        result = evaluate(
            Responses([None, "Answer: !!!"]),
            {"case": short_answer_task(name, [["!!!"], ["!!!"]])},
            bootstrap_iters=0,
            log_samples=True,
            verifyit_enabled=enabled,
        )
        for filter_name in ["strict_answer", "extract_answer"]:
            scores = [sample["exact_match"] for sample in result["samples"]["case"] if sample["filter"] == filter_name]
            assert scores == ([0, 0] if enabled else [0, 1])


class RollingLikelihoods(Responses):
    def loglikelihood_rolling(self, requests, **kwargs):
        return [self.responses[request.doc_id] for request in requests]


def rolling_task(contents):
    path = Path(__file__).parents[2] / "eval/lm_eval_tasks/uncheatable_eval/wikipedia_english.yaml"
    config = load_yaml(path, resolve_func=True)
    docs = [{"category": "wikipedia_english", "content": text} for text in contents]
    config.update(num_fewshot=0, custom_dataset=lambda **kwargs: DatasetDict(test=Dataset.from_list(docs)))
    return ConfigurableTask(config=config)


def test_rolling_diagnostics_preserve_corpus_weighting_and_default_reporting():
    contents = ["  café\t猫\n", "a b c d e"]
    results = [
        evaluate(
            RollingLikelihoods([-1.25, -17.5]),
            {"case": rolling_task(contents)},
            log_samples=True,
            verifyit_enabled=enabled,
        )
        for enabled in [False, True]
    ]
    assert results[0]["results"] == results[1]["results"]
    assert results[0]["samples"] == results[1]["samples"]
    metrics = results[1]["results"]["case"]
    assert metrics["word_perplexity,none"] == pytest.approx(math.exp(18.75 / 9))
    assert metrics["byte_perplexity,none"] == pytest.approx(math.exp(18.75 / 21))
    assert metrics["word_perplexity_stderr,none"] == "N/A"


@pytest.mark.parametrize("observation", [None, float("nan"), 1.0, -1000.0])
def test_invalid_rolling_observation_aborts_reporting_instead_of_emitting_favorable_metrics(observation):
    with pytest.raises(InvalidTask):
        evaluate(
            RollingLikelihoods([observation]),
            {"case": rolling_task(["text"])},
            verifyit_enabled=True,
        )


def test_rolling_category_callable_cannot_borrow_a_valid_source_namespace():
    task = rolling_task(["text"])
    original = task.config.process_docs
    alternate = original.__globals__["github_python"]
    task.config.process_docs = FunctionType(alternate.__code__, original.__globals__, original.__name__)
    task.dataset = DatasetDict(test=Dataset.from_list([{"category": "github_python", "content": "text"}]))
    with pytest.raises(InvalidTask, match="category"):
        evaluate(RollingLikelihoods([-1.0]), {"case": task}, verifyit_enabled=True)
