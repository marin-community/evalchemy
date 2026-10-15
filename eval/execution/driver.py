"""One evaluation driver over normalized source iterators."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from contextlib import aclosing
import logging
from typing import Any
from aiohttp import ClientSession, ClientTimeout

from eval.contracts.eval_spec import EvalSpec
from eval.contracts.sample_manifest import SampleCoverageError
from eval.contracts.task_outcome import (
    FailureCategory, FailurePhase, TaskOutcome, TaskRoute, TaskStatus, classify_task_exception,
)
from eval.execution.backend import LMBackend
from eval.execution.engine import CollectSamples, Evaluator, ItemFailure, ItemResult
from eval.execution.grading import Grader, Graders
from eval.execution.metrics import SourceMetrics
from eval.execution.run_spec import RunSpec
from eval.execution.judge import JudgeBackend
from eval.graders.answer_equivalence import JudgeConfig
from eval.graders.specs import GRADERS
from eval.sources.common import SourceOptions
from eval.sources.lm_eval import LMEvalMetrics, LMEvalSource
from eval.contracts.eval_spec import SourceRef
from eval.robust_api import completion_response_text_unavailable

logger = logging.getLogger(__name__)


@dataclass
class SourcePlan:
    name: str
    route: TaskRoute
    sources: tuple[Iterable[EvalSpec], ...]
    task_names: tuple[str, ...]
    reducer: Any
    graders: dict[str, Grader] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    judge_config: JudgeConfig | None = None
    expected_examples: int | None = None


def _interleave(plans: list[SourcePlan]) -> Iterator[EvalSpec]:
    active = deque((plan, iter(source)) for plan in plans for source in plan.sources)
    while active:
        plan, source = active.popleft()
        try:
            spec = next(source)
        except StopIteration:
            continue
        except Exception as exc:
            logger.error("Source %s failed while producing items: %s", plan.name, exc)
            plan.errors.append(exc)
            continue
        active.append((plan, source))
        yield spec


def _custom_items(benchmark, options):
    yield from benchmark.iter_specs(options)


def source_plans(tasks, routes, custom, native, run: RunSpec, model) -> list[SourcePlan]:
    plans = []
    imported = set()
    for task in tasks:
        route = TaskRoute(routes[task])
        if route is TaskRoute.CUSTOM:
            benchmark = custom.get_benchmark(task)
            description = benchmark.describe(task)
            if description is None:
                raise ValueError(f"source {task} must declare its metrics")
            source = SourceRef(benchmark.dataset_name, benchmark.dataset_split, benchmark.dataset_revision)
            options = SourceOptions(
                task, source, run.limit, run.max_length,
                run.max_tokens if run.max_tokens is not None else benchmark._evaluation_max_tokens,
                run.generation, run.seeds[0], run.num_samples, run.system_instruction,
                model.tok_encode,
            )
            reducer = SourceMetrics(
                repeated=False, pass_at_k=run.pass_at_k if run.num_samples > 1 else (),
                policy=description.reducers[0],
            )
            plans.append(SourcePlan(
                task, route, (_custom_items(benchmark, options),), (task,), reducer,
                judge_config=JudgeConfig.resolve(judge_model=run.judge_model or benchmark.judge_model)
                if benchmark.REQUIRES_JUDGE and not run.predict_only else None,
                expected_examples=description.n_attempted,
            ))
            continue
        loaded = native.load([task])
        sources = []
        graders = {}
        for name, definition in loaded["tasks"].items():
            if getattr(definition, "UNSAFE_CODE", False) and not run.confirm_run_unsafe_code:
                raise ValueError(f"task {name} requires --confirm_run_unsafe_code")
            source = LMEvalSource(
                name, definition, run.limit, run.num_fewshot, run.generation, run.seeds[3],
                run.apply_chat_template, run.system_instruction, run.fewshot_as_multiturn,
            )
            if name not in imported:
                sources.append(source)
                imported.add(name)
            graders[source.grader_name] = source.grader()
        reducer = LMEvalMetrics(loaded["tasks"], loaded["groups"])
        plans.append(SourcePlan(task, route, tuple(sources), tuple(loaded["tasks"]), reducer, graders))
    return plans


async def run_evaluations(model, plans: list[SourcePlan], *, workers: int, sink=None, checkpoints=None, predict_only=False) -> dict:
    samples = CollectSamples() if sink is None else sink
    definitions = dict(GRADERS)
    for plan in plans:
        definitions.update(plan.graders)
    grading = Graders(definitions, workers)
    by_task = defaultdict(list)
    for plan in plans:
        for task in plan.task_names:
            by_task[task].append(plan)
    scored: dict[str, set[str]] = {task: set() for task in by_task}
    try:
        async with LMBackend(model) as backend, ClientSession(timeout=ClientTimeout(total=None)) as judge_session:
            judges = {
                task: JudgeBackend(plan.judge_config, judge_session)
                for plan in plans if plan.judge_config is not None for task in plan.task_names
            }
            engine = Evaluator(backend, grading, samples, workers=workers, checkpoints=checkpoints, judges=judges,
                               predict_only=predict_only)
            async with aclosing(engine.run(_interleave(plans))) as stream:
                async for completed in stream:
                    task_plans = by_task[completed.job.task]
                    if completed.error is not None:
                        logger.error("Evaluation %s item %s failed: %s", completed.job.task, completed.job.source_id, completed.error)
                        for plan in task_plans:
                            plan.errors.append(completed.error)
                        continue
                    result: ItemResult = completed.result
                    if not predict_only:
                        scored[result.spec.task].add(result.record["sample_id"])
                        for plan in task_plans:
                            plan.reducer.add(result)
    finally:
        grading.close()
    result = {"results": {}, "task_outcomes": {}}
    for plan in plans:
        manifests = [engine.manifests[task] for task in plan.task_names if task in engine.manifests]
        expected = sum(manifest.expected_sample_count for manifest in manifests)
        generated = sum(manifest.generated_sample_count for manifest in manifests)
        score_count = sum(len(scored[task]) for task in plan.task_names)
        if plan.expected_examples is not None and expected != plan.expected_examples:
            plan.errors.append(SampleCoverageError(f"source declared {plan.expected_examples} examples but yielded {expected}"))
        for manifest in manifests:
            try:
                manifest.validate_generated()
                if not predict_only:
                    manifest.mark_scored(len(scored[manifest.task_name]))
                    manifest.validate_scored()
            except SampleCoverageError as exc:
                plan.errors.append(exc)
        if not expected:
            plan.errors.append(SampleCoverageError("source produced no evaluation items"))
        failures = Counter()
        response_summary = Counter()
        for task in plan.task_names:
            failures.update(engine.failures[task])
            response_summary.update(engine.response_summary[task])
        if completion_response_text_unavailable(response_summary):
            plan.errors.append(ItemFailure(FailurePhase.GENERATION, ValueError("at least half of chat completions contain no scorer text"), FailureCategory.MALFORMED_MODEL_RESPONSE))
        if plan.errors:
            error = plan.errors[0]
            category = error.category if isinstance(error, ItemFailure) else FailureCategory.INCOMPLETE_EVALUATION if isinstance(error, SampleCoverageError) else classify_task_exception(FailurePhase.EVALUATION, error)
            outcome = TaskOutcome.failed(plan.name, plan.route, category, str(error), exception=error)
            outcome = replace(outcome, expected_count=expected, generated_count=generated, scored_count=score_count,
                              failure_counts=failures, completion_response_summary=response_summary)
        elif predict_only:
            outcome = TaskOutcome(plan.name, plan.route, TaskStatus.EXPORTED, {}, expected, generated, None,
                                  failure_counts=failures, completion_response_summary=response_summary)
        else:
            document = plan.reducer.finish(plan.name)
            for key, value in document.items():
                result.setdefault(key, {}).update(value)
            metrics = document["results"].get(plan.name, document.get("groups", {}).get(plan.name, {}))
            outcome = TaskOutcome(
                plan.name, plan.route, TaskStatus.SUCCEEDED, metrics, expected, generated, score_count,
                failure_counts=failures, completion_response_summary=response_summary,
            )
        result["task_outcomes"][plan.name] = outcome.to_dict()
    if sink is None:
        result["samples"] = samples.samples
    return result
