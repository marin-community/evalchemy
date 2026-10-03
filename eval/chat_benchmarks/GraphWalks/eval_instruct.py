"""Revision-pinned GraphWalks with model-specific context selection and set-F1 scoring."""

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from types import MappingProxyType

from datasets import load_dataset
from lm_eval.api.instance import Instance

from eval.completion_response import FailedGeneration
from eval.contracts.benchmark_metadata import MetricKind
from eval.contracts.sample_results import record_sample_metrics
from eval.limits import endpoint_prompt_token_count, require_context_length
from evalchemy_config.limits import MAX_OUTPUT_ALIASES, resolve_limit
from eval.task import BaseBenchmark

GRAPHWALKS_DATASET = "openai/graphwalks"
GRAPHWALKS_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
_FINAL_ANSWER_PREFIX = "Final Answer:"
_FINAL_ANSWER = re.compile(r"\[.*\]")
_CONTEXT_MARGIN = 64
_PREFIX_TOKEN_MARGIN = 256
_PREFIX_CHARS_PER_TOKEN = 8
_MIN_OUTPUT_TOKENS = 4096
_REASONING_RESERVE = 4096
_OUTPUT_BUDGET_MULTIPLIER = 2


@dataclass(frozen=True)
class GraphWalksGrade:
    """Extracted answer and its set-based scores."""

    scores: dict[str, float]
    extracted: list[str]
    failed_to_parse: bool


def extract_answer(response: str) -> tuple[list[str], bool]:
    """Extract the last-line answer and report whether parsing failed."""
    line = response.rstrip().split("\n")[-1]
    if _FINAL_ANSWER_PREFIX not in line:
        return [], True
    match = _FINAL_ANSWER.search(line)
    if match is None:
        return [], True
    return [node.strip() for node in match.group(0).strip("[]").split(",") if node.strip()], False


def grade_answer(response: str, answer_nodes: tuple[str, ...]) -> GraphWalksGrade:
    """Score a predicted node set using GraphWalks precision, recall, and F1."""
    extracted, failed_to_parse = extract_answer(response)
    predicted = set(extracted)
    truth = set(answer_nodes)
    if failed_to_parse:
        precision = recall = f1 = 0.0
    elif not truth and not predicted:
        precision = recall = f1 = 1.0
    else:
        overlap = len(predicted & truth)
        recall = overlap / len(truth) if truth else 0.0
        precision = overlap / len(predicted) if predicted else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return GraphWalksGrade(
        scores={
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "exact_match": float(predicted == truth and not failed_to_parse),
        },
        extracted=extracted,
        failed_to_parse=failed_to_parse,
    )


class GraphWalksBenchmark(BaseBenchmark):
    """Score the context-eligible subset while retaining full benchmark coverage."""

    METRICS = ("f1", "precision", "recall", "exact_match")
    PRIMARY_METRIC = "f1"
    METRIC_NAME_OVERRIDES = MappingProxyType({"exact_match": "exact_match"})
    METRIC_KIND_OVERRIDES = MappingProxyType({"exact_match": MetricKind.BINARY})

    def __init__(self, max_tokens=131072, logger=None, system_instruction=None, cache_dir=None):
        if system_instruction is not None:
            raise ValueError("GraphWalks uses the canonical dataset prompt without a system instruction")
        super().__init__(logger=logger)
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.selection = None

    def benchmark_size(self):
        return None if self.selection is None else self.selection["n_benchmark"]

    def describe(self, task_name=None):
        metadata = super().describe(task_name)
        if self.selection is None:
            return metadata
        return replace(metadata, n_attempted=self.selection["n_attempted"])

    def request_max_tokens(self, instance, output_cap):
        required = instance.doc["max_output_tokens"]
        if output_cap is not None and required > output_cap:
            raise ValueError("GraphWalks request exceeds the configured output cap")
        return required

    def _load_rows(self):
        return load_dataset(GRAPHWALKS_DATASET, split="train", revision=GRAPHWALKS_REVISION, cache_dir=self.cache_dir)

    def generate_responses(self, model):
        context = require_context_length(self._evaluation_max_length, task_name=self.benchmark_name)
        if model.tokenizer is None:
            raise ValueError("GraphWalks requires the served model's tokenizer")
        override_cap = resolve_limit(
            "max_tokens",
            [
                (alias, self._evaluation_gen_kwargs[alias])
                for alias in MAX_OUTPUT_ALIASES
                if alias in self._evaluation_gen_kwargs
            ],
        )
        cap = override_cap or self._evaluation_max_tokens or self.max_tokens
        if context <= 8192 + _CONTEXT_MARGIN or cap < 8192:
            raise ValueError("GraphWalks requires room for the prompt and at least 8192 output tokens")
        template_kwargs = getattr(model, "_evalchemy_chat_template_kwargs", {})
        dataset = self._load_rows()
        selected = []
        context_skips = Counter()
        output_skips = Counter()
        capped = 0
        for index, row in enumerate(dataset):
            if self.evaluation_limit is not None and len(selected) >= self.evaluation_limit:
                capped = len(dataset) - index
                break
            answer = _FINAL_ANSWER_PREFIX + " [" + ", ".join(row["answer_nodes"]) + "]"
            answer_tokens = len(model.tokenizer.encode(answer, add_special_tokens=False))
            budget = _OUTPUT_BUDGET_MULTIPLIER * max(_MIN_OUTPUT_TOKENS, answer_tokens + _REASONING_RESERVE)
            if budget > cap:
                output_skips[row["problem_type"]] += 1
                continue
            prompt_budget = context - budget - _CONTEXT_MARGIN
            if prompt_budget <= 0:
                context_skips[row["problem_type"]] += 1
                continue
            messages = [{"role": "user", "content": row["prompt"]}]
            if row["prompt_chars"] > prompt_budget * _PREFIX_CHARS_PER_TOKEN:
                prefix = [{"role": "user", "content": row["prompt"][: prompt_budget * _PREFIX_CHARS_PER_TOKEN]}]
                prefix_tokens = endpoint_prompt_token_count(
                    model.tokenizer, prefix, chat_template_kwargs=template_kwargs
                )
                if prefix_tokens > prompt_budget + _PREFIX_TOKEN_MARGIN:
                    context_skips[row["problem_type"]] += 1
                    continue
            tokens = endpoint_prompt_token_count(model.tokenizer, messages, chat_template_kwargs=template_kwargs)
            if tokens > prompt_budget:
                context_skips[row["problem_type"]] += 1
                continue
            example = dict(row)
            example.update(
                source_index=index,
                answer=json.dumps(row["answer_nodes"]),
                prompt_tokens=tokens,
                max_output_tokens=budget,
            )
            payload = self._prepare_messages(messages, model)
            selected.append((example, payload))
        self.selection = {
            "dataset": GRAPHWALKS_DATASET,
            "revision": GRAPHWALKS_REVISION,
            "n_benchmark": len(dataset),
            "n_attempted": len(selected),
            "skipped_context_by_type": dict(context_skips),
            "skipped_output_cap_by_type": dict(output_skips),
            "not_inspected_after_limit": capped,
            "max_model_len": context,
            "max_output_tokens": cap,
            "chat_template_kwargs": dict(template_kwargs),
            "output_budget_policy": "2 * max(4096, tokenized_gold_list_length + 4096)",
        }
        if not selected:
            raise ValueError("No GraphWalks examples fit the served context and output cap")
        instances = [
            Instance(
                "generate_until",
                example,
                (payload, {"max_new_tokens": example["max_output_tokens"], "temperature": 0.0}),
                example["source_index"],
            )
            for example, payload in selected
        ]
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None
        examples = []
        for (example, _), output in zip(selected, outputs):
            example["model_output"] = output
            examples.append(example)
        return {"examples": examples, "selection": self.selection}

    def evaluate_responses(self, results):
        if results is None:
            return None
        examples = results["examples"]
        totals = {metric: [] for metric in self.METRICS}
        errors = Counter()
        unanswered = 0
        for example in examples:
            output = example["model_output"]
            if isinstance(output, FailedGeneration):
                errors[output.failure_category] += 1
                record_sample_metrics(example, **{metric: 0.0 for metric in self.METRICS})
                continue
            grade = grade_answer(str(output), tuple(example["answer_nodes"]))
            example["model_answer"] = json.dumps(grade.extracted)
            example["answer_extraction_error"] = "failed_to_parse" if grade.failed_to_parse else None
            record_sample_metrics(example, **grade.scores)
            unanswered += grade.failed_to_parse
            for metric, score in grade.scores.items():
                totals[metric].append(score)
        count = len(totals["f1"])
        if not count or count / len(examples) < 0.9:
            raise ValueError(f"GraphWalks scored {count}/{len(examples)} attempted examples, below 90%")
        stderr = statistics.stdev(totals["f1"]) / math.sqrt(count) if count > 1 else 0.0
        return {
            **{metric: statistics.mean(scores) for metric, scores in totals.items()},
            "f1_stderr": stderr,
            "num_total": count,
            "scored_count": count,
            "n_unanswered": unanswered,
            "errors": dict(errors),
            "selection": results["selection"],
            "examples": examples,
        }

    def _sample_gen_kwargs(self, example):
        overrides = {key: value for key, value in self._evaluation_gen_kwargs.items() if key not in MAX_OUTPUT_ALIASES}
        return {"max_new_tokens": example["max_output_tokens"], "temperature": 0.0, **overrides}

    def to_samples(self, generation_result, scored_result):
        samples = super().to_samples(generation_result, scored_result)
        for sample, example in zip(samples, generation_result["examples"]):
            sample["doc_id"] = example["source_index"]
        return samples
