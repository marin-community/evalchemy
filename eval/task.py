import importlib.util
import inspect
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from itertools import islice
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar, Union

import lm_eval.models as lm_eval_models
import numpy as np

try:  # torch is optional: endpoint-only installs (no [vllm]/[benchmarks]) run torch-free
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    torch = None
    dist = None

# The vLLM model class is only importable with the [vllm] extra. Resolve it defensively so
# the endpoint path (local-completions / curator, no vllm) does not AttributeError on the
# `isinstance(model, VLLM)` checks below; isinstance(x, ()) is always False.
try:
    from lm_eval.models.vllm_causallms import VLLM as _VLLM
except Exception:
    _VLLM = ()
# Force-import the OpenAI-completions submodule so `lm_eval_models.openai_completions.*` in
# `_normalize_model_args` resolves regardless of which model registered: `local-completions`
# imports it as a side effect, `curator` does not. Base lm-eval ([api]); no extra required.
import lm_eval.models.openai_completions  # noqa: F401,E402
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from evalchemy_config.limits import MAX_OUTPUT_ALIASES, resolve_limit

from eval.contracts.benchmark_metadata import (
    BenchmarkMetadata,
    MetricKind,
    SourceMetric,
    resolve_metric_metadata,
)
from eval.contracts.conformance import find_custom_benchmark_classes, validate_custom_benchmark_class
from eval.contracts.grading import GraderExecutionMode
from eval.contracts.preflight import ResourceRequirement, TaskPreparation, prepare_task, validate_model_request
from eval.contracts.prompt_length import load_prompt_lengths, resolve_task_max_tokens
from eval.contracts.sample_manifest import (
    DEFAULT_SAMPLE_NAMESPACE,
    SampleEntry,
    SampleManifest,
    SampleRequest,
    canonical_json_identity,
)
from eval.contracts.sample_results import (
    SAMPLE_METRICS_ANNOTATION,
    record_sample_metrics,
    sample_metric_fields,
)
from eval.contracts.task_outcome import TaskRoute
from eval.robust_api import parse_generation_overrides
from eval.passk import estimate_pass_at_k


_Sample = TypeVar("_Sample")


class BaseBenchmark(ABC):
    """Abstract base class for implementing LLM evaluation benchmarks."""

    RESOURCE_REQUIREMENTS: tuple[ResourceRequirement, ...] = ()
    GRADER_EXECUTION_MODE = GraderExecutionMode.SERIAL
    METRICS: tuple[str, ...] = ()
    PRIMARY_METRIC: str | None = None
    METRIC_NAME_OVERRIDES: Mapping[str, str] = MappingProxyType({})
    METRIC_KIND_OVERRIDES: Mapping[str, MetricKind | str] = MappingProxyType({})

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Union[str, List[int]]] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.system_instruction = system_instruction
        # Native pass@k controls (Stage 2b). ``num_samples=1`` (the default)
        # is a strict no-op: every benchmark takes its current single-sample
        # path and produces byte-identical output. ``pass_at_k`` is only
        # consulted when ``num_samples > 1``.
        from eval.passk import parse_pass_at_k

        self.num_samples = int(num_samples) if num_samples else 1
        self.pass_at_k = parse_pass_at_k(pass_at_k)
        # Resume manager (Stage 3a). ``None`` (the default) is a strict no-op:
        # ``compute`` takes its current generate-everything path and produces
        # byte-identical output. Wired by the driver (Stage 4) via
        # ``attach_resume_manager``; until then nothing attaches it.
        self._resume_manager = None
        # Set by the native pass@k batch path to suspend the per-(problem,repeat)
        # resume wrap on its inner ``compute`` calls (Stage 3c).
        self._suspend_resume = False
        # The driver sets these once after benchmark construction.  Subclasses
        # predate the common limit contract and commonly keep their own
        # ``max_tokens`` / ``max_new_tokens`` fields, so request-time enforcement
        # below is the final, framework-wide guard against a benchmark silently
        # using a different output budget.
        self._evaluation_max_length: Optional[int] = None
        self._evaluation_max_tokens: Optional[int] = None
        self._evaluation_gen_kwargs: Dict[str, Any] = {}
        self._evaluation_limit: Optional[int] = None
        self._limited_sample_ids: Dict[str, set[str]] = {}
        self._sample_manifest = SampleManifest(self.benchmark_name)

    @property
    def benchmark_name(self) -> str:
        """Return the canonical task name derived from the benchmark class."""
        return self.__class__.__name__.replace("Benchmark", "")

    @property
    def sample_manifest(self) -> SampleManifest:
        """Return this benchmark run's shared sample lifecycle ledger."""
        return self._sample_manifest

    def prepare(self, task_name: str | None = None) -> TaskPreparation:
        """Validate static resources and loaded data without contacting a model."""
        name = task_name or self.benchmark_name
        return prepare_task(
            name,
            TaskRoute.CUSTOM,
            self.RESOURCE_REQUIREMENTS,
            self.validate_prepared_data,
        )

    def benchmark_size(self) -> int | None:
        """Return the prepared pre-limit item count, or ``None`` when unknown."""
        return None

    def benchmark_metrics(self) -> tuple[str, ...]:
        """Return source metrics for this benchmark's resolved run configuration."""
        if self.num_samples > 1:
            return tuple(f"pass@{k}" for k in self.pass_at_k if k <= self.num_samples)
        return self.METRICS

    def benchmark_primary_metric(self) -> str | None:
        """Return the source spelling of the preferred headline metric."""
        if self.num_samples > 1:
            metrics = self.benchmark_metrics()
            return "pass@1" if "pass@1" in metrics else (metrics[0] if metrics else None)
        return self.PRIMARY_METRIC

    def describe(self, task_name: str | None = None) -> BenchmarkMetadata | None:
        """Return canonical metrics and coverage, or ``None`` when metrics are undeclared."""
        source_metrics = self.benchmark_metrics()
        if not source_metrics:
            return None
        name = task_name or self.benchmark_name
        metrics, primary = resolve_metric_metadata(
            tuple(SourceMetric(metric) for metric in source_metrics),
            primary_metric=self.benchmark_primary_metric(),
            name_overrides=self.METRIC_NAME_OVERRIDES,
            kind_overrides=self.METRIC_KIND_OVERRIDES,
        )
        n_benchmark = self.benchmark_size()
        n_attempted = (
            min(self.evaluation_limit, n_benchmark)
            if self.evaluation_limit is not None and n_benchmark is not None
            else n_benchmark
        )
        primary_kind = next(metric.kind for metric in metrics if metric.name == primary)
        return BenchmarkMetadata(name, primary, primary_kind, metrics, n_benchmark, n_attempted)

    def validate_prepared_data(self) -> None:
        """Hook for dataset shape and representative-request validation."""

    def set_evaluation_limits(
        self,
        *,
        max_length: Optional[int] = None,
        max_tokens: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        """Attach Evalchemy's resolved limits to this custom benchmark.

        ``max_length`` is already supplied to the LM adapter through
        ``model_args``.  MMLU-Pro additionally owns its prompt-budgeting logic,
        so synchronize its legacy field here.  ``max_tokens`` is forced on every
        generated ``Instance`` in :meth:`_normalize_model_args`, including
        benchmarks with a hard-coded per-task default. ``limit`` gives custom
        benchmarks the same positive sample cap as native lm-eval tasks.
        """
        self._evaluation_max_length = max_length
        self._evaluation_max_tokens = max_tokens
        self._evaluation_limit = limit if limit is not None and limit > 0 else None
        self._limited_sample_ids.clear()
        if max_length is not None and hasattr(self, "max_model_length"):
            self.max_model_length = max_length
        if max_tokens is not None:
            # Keep the legacy fields coherent for prompt constructors that use
            # their own stored value before they create an Instance.
            if hasattr(self, "max_tokens"):
                self.max_tokens = max_tokens
            if hasattr(self, "max_new_tokens"):
                self.max_new_tokens = max_tokens
            config = getattr(self, "config", None)
            if config is not None and hasattr(config, "max_new_token"):
                config.max_new_token = max_tokens

    def set_evaluation_generation_kwargs(self, gen_kwargs: Dict[str, Any]) -> None:
        """Store caller settings for request-time normalization."""
        self._evaluation_gen_kwargs = parse_generation_overrides(gen_kwargs)

    @property
    def evaluation_limit(self) -> Optional[int]:
        """Return the normalized sample cap for this custom benchmark."""
        return self._evaluation_limit

    def limit_samples(self, samples: Iterable[_Sample]) -> List[_Sample]:
        """Materialize at most the configured number of source samples."""
        if self.evaluation_limit is None:
            return list(samples)
        return list(islice(samples, self.evaluation_limit))

    def _limit_instances(self, inputs: List[Instance], sample_namespace: str) -> List[Instance]:
        """Enforce the sample cap at the common custom-benchmark inference boundary.

        Selection is tracked per namespace and source identity so chunked callers
        cannot exceed the cap, while repeated generations of an already-selected
        sample (for example pass@k) remain valid.
        """
        if self.evaluation_limit is None:
            return inputs

        selected_ids = self._limited_sample_ids.setdefault(sample_namespace, set())
        limited_inputs = []
        for instance in inputs:
            source_id = canonical_json_identity(instance.idx)
            if source_id not in selected_ids:
                if len(selected_ids) >= self.evaluation_limit:
                    continue
                selected_ids.add(source_id)
            limited_inputs.append(instance)
        return limited_inputs

    def attach_resume_manager(self, manager) -> None:
        """Attach a ResumeManager so ``compute`` skips already-done problems.

        A ``None`` manager (the default) leaves ``compute`` byte-identical to
        today (global invariant #1). The driver constructs the manager from the
        run fingerprint and attaches it here (Stage 4); Stage 3a only wires the
        consumption side.
        """
        self._resume_manager = manager

    def generate_n_samples(
        self,
        model: LM,
        build_instances: Callable[[int, List[int]], List[Instance]],
        num_samples: Optional[int] = None,
    ) -> List[List[str]]:
        """Generic "N completions per problem" generator (Stage 2b).

        Generalizes the per-(problem, sample) scaffold that AIME24/AMC23 hand-
        rolled (``for i in range(self.n_repeat): ... compute(...)``) up into the
        base so any benchmark can request ``num_samples`` completions per problem
        and grade each with its own grader.

        ``build_instances(sample_idx, seed)`` must return one ``Instance`` per
        problem (in problem order) for that sample pass; it owns the prompt,
        decoding params, ``repeat_idx`` and metadata exactly as the subclass
        does today. This method loops over the samples, calls
        ``self.compute(...)`` once per sample pass (each problem keeps its full
        per-pass batch — no n-split), and on rank 0 returns a list (one entry
        per problem) of lists (one completion per sample).

        Non-primary ranks get ``None`` propagated from ``compute``/all_gather.
        """
        n = int(num_samples if num_samples is not None else self.num_samples)
        base_seed = getattr(self, "seed", [0, 1234, 1234, 1234])
        all_outputs: List[List[str]] = []
        for sample_idx in range(n):
            seed = [s + sample_idx for s in base_seed]
            instances = build_instances(sample_idx, seed)
            outputs = self.compute(model, instances)
            all_outputs.append(outputs)
        if model.rank != 0:
            return None
        # transpose: all_outputs[sample][problem] -> per_problem[problem][sample]
        return [list(per_problem) for per_problem in zip(*all_outputs)]

    def generate_n_samples_batched(
        self,
        model: LM,
        build_instances: Callable[[int, List[int]], List[Instance]],
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[str]]:
        """Resume-aware native pass@k generation in per-problem batches (Stage 3c).

        Same contract as :meth:`generate_n_samples` (return, on rank 0, a list with
        one entry per problem, each a list of ``num_samples`` completions) but the
        unit of resume is a **problem-batch** of size ``B`` (``{task, batch_idx}``,
        stage-0 decision #4). For each batch the full ``num_samples`` are generated
        per problem (no n-split), so a fresh run and a resumed run produce identical
        per-(problem, sample) outputs by construction (global invariant #7).

        Flag-off invariant: when no manager is attached (the default), this delegates
        to :meth:`generate_n_samples` and is **byte-identical to the Stage-2b output**.
        With a manager: completed batches are ``should_skip``-ped and their per-
        (problem, sample) outputs ``restore``-d from the manifest; only the remaining
        batches are regenerated and ``record``-ed as each finishes; the union is
        re-assembled in problem order so aggregation is unchanged.

        ``build_instances(sample_idx, seed)`` returns one ``Instance`` per problem
        (full problem list, in problem order) for that sample pass — exactly the
        callback :meth:`generate_n_samples` takes. This method slices it per batch.
        """
        manager = getattr(self, "_resume_manager", None)
        n = int(num_samples if num_samples is not None else self.num_samples)
        # No manager (or off-mode) -> the Stage-2b path verbatim (byte-identical).
        if manager is None:
            return self.generate_n_samples(model, build_instances, n)

        task_name = self.benchmark_name
        B = int(batch_size) if batch_size else self._passk_batch_size()

        # Build the full per-sample instance lists once; slice per batch below. This
        # matches generate_n_samples' instance construction exactly (same seeds, same
        # repeat_idx) so each (problem, sample) is identical regardless of batching.
        base_seed = getattr(self, "seed", [0, 1234, 1234, 1234])
        per_sample_instances: List[List[Instance]] = []
        for sample_idx in range(n):
            seed = [s + sample_idx for s in base_seed]
            sample_instances = build_instances(sample_idx, seed)
            sample_instances = self._limit_instances(sample_instances, DEFAULT_SAMPLE_NAMESPACE)
            for ordinal, instance in enumerate(sample_instances):
                instance.sample_ordinal = ordinal
                instance.sample_repeat = getattr(instance, "repeat_idx", sample_idx)
            per_sample_instances.append(sample_instances)

        passk_entries = [
            self.sample_manifest.plan_batch(
                [
                    SampleRequest(
                        source_id=instance.idx,
                        ordinal=instance.sample_ordinal,
                        repeat=instance.sample_repeat,
                    )
                    for instance in sample_instances
                ]
            )
            for sample_idx, sample_instances in enumerate(per_sample_instances)
        ]

        num_problems = len(per_sample_instances[0]) if per_sample_instances else 0
        # batch_idx -> list of problem indices it covers
        batches = [list(range(i, min(i + B, num_problems))) for i in range(0, num_problems, B)]

        manager.decide()  # fresh / resume / refuse (loud refuse on material delta, inv #3)
        restored = manager.restore()  # {unit_key: payload} for completed batches
        from eval.resume import canonical_unit_key

        # per_problem_outputs[problem_idx] -> list of `n` completions
        per_problem_outputs: List[Optional[List[str]]] = [None] * num_problems
        skipped = 0
        for batch_idx, problem_idxs in enumerate(batches):
            unit = {"task": task_name, "batch_idx": batch_idx}
            batch_entries = [
                passk_entries[sample_idx][problem_idx]
                for sample_idx in range(n)
                for problem_idx in problem_idxs
            ]
            if manager.should_skip(unit):
                payload = restored[canonical_unit_key(unit)]
                self.sample_manifest.validate_prior_entries([payload], batch_entries)
                # payload["outputs"] is keyed by problem index (as strings in JSON)
                stored = payload["outputs"]
                for pidx in problem_idxs:
                    per_problem_outputs[pidx] = list(stored[str(pidx)])
                skipped += 1
                continue
            # Generate the full num_samples for just this batch's problems. Suspend
            # the per-(problem,repeat) resume wrap on the inner compute so pass@k
            # records ONLY the {task, batch_idx} unit (no double-recording).
            batch_per_problem: Dict[int, List[str]] = {pidx: [] for pidx in problem_idxs}
            self._suspend_resume = True
            try:
                for sample_idx in range(n):
                    sub_instances = [per_sample_instances[sample_idx][pidx] for pidx in problem_idxs]
                    outputs = self.compute(model, sub_instances)
                    if model.rank == 0:
                        for pidx, out in zip(problem_idxs, outputs):
                            batch_per_problem[pidx].append(out)
            finally:
                self._suspend_resume = False
            if model.rank == 0:
                for pidx in problem_idxs:
                    per_problem_outputs[pidx] = batch_per_problem[pidx]
                manager.record(
                    unit,
                    {
                        "outputs": {
                            str(pidx): batch_per_problem[pidx]
                            for pidx in problem_idxs
                        },
                        "samples": [entry.to_dict() for entry in batch_entries],
                    },
                )

        if skipped:
            self.logger.info(
                f"resume[{task_name}] rank={getattr(model, 'rank', 0)}: skipped {skipped} done "
                f"batches (B={B}), generating {len(batches) - skipped} remaining"
            )

        manager.finalize()
        if model.rank != 0:
            return None
        completed = [list(problem_outputs) for problem_outputs in per_problem_outputs]
        for sample_idx, entries in enumerate(passk_entries):
            self.sample_manifest.mark_generated(
                entries,
                [problem_outputs[sample_idx] for problem_outputs in completed],
            )
        return completed

    def _passk_batch_size(self) -> int:
        """Fingerprinted pass@k problem-batch size ``B``. Defaults to all problems in one batch.

        A subclass / driver may set ``self.passk_batch_size``; absent that, the whole
        problem list is one batch (equivalent to the Stage-2b single-pass behavior, just
        resume-checkpointed once). ``B`` is a fingerprint input (decision #4) so changing
        it refuses a resume.
        """
        b = getattr(self, "passk_batch_size", None)
        if b:
            return int(b)
        return 1 << 30  # effectively "all problems"

    def aggregate_pass_at_k(
        self,
        num_correct: List[int],
        num_samples: Optional[int] = None,
        ks: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Aggregate per-problem correct counts into a {pass@k: mean} table.

        Thin wrapper over the shared estimator in ``eval/passk.py`` so there is
        exactly one definition of the unbiased estimator in the tree.
        """
        from eval.passk import aggregate_pass_at_k as _agg

        n = int(num_samples if num_samples is not None else self.num_samples)
        return _agg(n, num_correct, ks if ks is not None else self.pass_at_k)

    def record_repeated_accuracy(
        self,
        examples: Sequence[Dict[str, Any]],
        correct_by_repeat: Sequence[Sequence[Any]],
    ) -> None:
        """Record each example's accuracy averaged over its repeated completions.

        Args:
            examples: The graded examples, annotated in place.
            correct_by_repeat: One score list per repetition, each ordered like
                ``examples``. Booleans and partial-credit scores are both accepted.
        """
        if not correct_by_repeat:
            raise ValueError(f"{self.benchmark_name}: cannot record per-sample accuracy without a repetition")
        for index, example in enumerate(examples):
            scores = [float(repeat[index]) for repeat in correct_by_repeat]
            record_sample_metrics(example, accuracy=sum(scores) / len(scores))

    def record_pass_at_k_metrics(
        self,
        examples: Sequence[Dict[str, Any]],
        num_correct: List[int],
        num_samples: Optional[int] = None,
        ks: Optional[List[int]] = None,
    ) -> None:
        """Record each problem's unbiased pass@k estimates as its per-sample metrics.

        The per-problem estimates are the terms ``aggregate_pass_at_k`` averages,
        so a sample artifact and the reported table stay reconcilable.
        """
        n = int(num_samples if num_samples is not None else self.num_samples)
        reportable = [k for k in (ks if ks is not None else self.pass_at_k) if k <= n]
        if not reportable:
            raise ValueError(f"{self.benchmark_name}: no requested pass@k is reportable from {n} samples")
        estimates = {f"pass_at_{k}": estimate_pass_at_k(n, num_correct, k) for k in reportable}
        for index, example in enumerate(examples):
            record_sample_metrics(example, **{name: values[index] for name, values in estimates.items()})

    def _normalize_model_args(self, model: LM, instances: List[Instance]) -> List[Instance]:
        overrides = dict(self._evaluation_gen_kwargs)
        override_cap = resolve_limit(
            "max_tokens",
            [(f"gen_kwargs.{alias}", overrides.pop(alias)) for alias in MAX_OUTPUT_ALIASES if alias in overrides],
        )
        output_cap = override_cap if override_cap is not None else self._evaluation_max_tokens
        for instance in instances:
            if output_cap is not None:
                # The request is the last common point all custom benchmarks
                # traverse.  Discard every backend spelling before assigning the
                # canonical output cap so a task-local default cannot win.
                for alias in MAX_OUTPUT_ALIASES:
                    instance.args[1].pop(alias, None)
                instance.args[1]["max_new_tokens"] = output_cap
            seeds = None
            if "seed" in instance.args[1]:
                seeds = instance.args[1]["seed"]

                random.seed(seeds[0])
                np.random.seed(seeds[1])
                if torch is not None:
                    torch.manual_seed(seeds[2])

                if instance.args[1].get("do_sample", True) is False:
                    # do_sample=False means greedy decoding. API models have no do_sample
                    # knob, so express the intent as temperature 0 -- otherwise a leftover
                    # sampling temperature makes OpenAI-compatible servers sample, and a
                    # temperature>0 request that carries a seed (lm-eval injects its own
                    # into every payload) is rejected outright by the JAX/TPU vLLM backend
                    # ("JAX does not support per-request seed"). The per-instance seed list
                    # is likewise not a valid API request seed, so drop it.
                    del instance.args[1]["seed"]
                    instance.args[1]["temperature"] = 0.0
                elif isinstance(model, lm_eval_models.openai_completions.LocalCompletionsAPI):
                    # LocalCompletionsAPI is the root of all four OpenAI-compatible API model
                    # classes (Local/OpenAI x completions/chat); the OpenAI* classes are the
                    # SUBCLASSES, so checking those two alone silently routes local-* endpoints
                    # into the Huggingface branch.
                    instance.args[1]["seed"] = seeds[0] if "seed" in instance.args[1] else None
                elif isinstance(model, _VLLM) or "UploadInstancesToHF" in model.__class__.__name__:
                    instance.args[1]["seed"] = seeds[0] if "seed" in instance.args[1] else None
                else:  # Huggingface does not support seed
                    _ = instance.args[1].pop("seed") if "seed" in instance.args[1] else None
            if "max_new_tokens" in instance.args[1]:
                max_new_tokens = instance.args[1].pop("max_new_tokens")
                if isinstance(model, lm_eval_models.openai_completions.LocalCompletionsAPI):
                    instance.args[1]["max_tokens"] = max_new_tokens
                    if "4o" in model.model:
                        instance.args[1]["max_tokens"] = min(max_new_tokens, 16384)
                elif isinstance(model, _VLLM):
                    instance.args[1]["max_gen_toks"] = max_new_tokens
                else:  # Huggingface
                    instance.args[1]["max_new_tokens"] = max_new_tokens
            if overrides.get("do_sample") is True and "temperature" not in overrides:
                instance.args[1].pop("temperature", None)
            instance.args[1].update(overrides)
        return instances

    def _prepare_messages(
        self, messages: List[Dict[str, str]], model: Optional[LM] = None
    ) -> Union[List[Dict[str, str]], str]:
        """Prepare messages with system instruction if available and apply chat template if model is provided.

        Args:
            messages: List of message dictionaries
            model: Optional language model instance for applying chat template

        Returns:
            If model is provided, returns the templated string. Otherwise returns the prepared message list.
        """
        if self.system_instruction:
            messages.insert(0, {"role": "system", "content": self.system_instruction})

        if model is not None:
            return model.apply_chat_template(messages)

        return messages

    def _unit_key(self, task_name: str, instance: Instance) -> Dict[str, Any]:
        """Return the legacy resume key, preserving opaque source IDs.

        New records use :class:`SampleEntry` IDs. This helper remains for reading
        existing manifests and intentionally no longer coerces IDs to integers.
        """
        key: Dict[str, Any] = {"task": task_name, "problem_idx": instance.idx}
        repeat_idx = getattr(instance, "repeat_idx", None)
        if repeat_idx is not None:
            key["repeat_idx"] = repeat_idx
        return key

    def compute(
        self,
        model: LM,
        inputs: List[Instance],
        do_slice: bool = True,
        sample_namespace: str = DEFAULT_SAMPLE_NAMESPACE,
    ) -> List[str]:
        inputs = self._limit_instances(inputs, sample_namespace)
        inputs = self._normalize_model_args(model, inputs)
        if inputs:
            # Prompt rendering depends on the selected model, so keep this bounded
            # representative check separate from static preflight while still failing
            # before the first generation request is sent.
            validate_model_request(inputs[0])

        # Add task_name to each instance
        task_name = self.benchmark_name
        for instance in inputs:
            instance.task_name = task_name

        world_size = int(getattr(model, "world_size", 1))
        entries = self.sample_manifest.plan_batch(
            [
                SampleRequest(
                    source_id=instance.idx,
                    ordinal=getattr(instance, "sample_ordinal", ordinal),
                    namespace=sample_namespace,
                    shard=ordinal % world_size if world_size > 1 and do_slice else None,
                    repeat=getattr(
                        instance,
                        "repeat_idx",
                        getattr(instance, "sample_repeat", None),
                    ),
                )
                for ordinal, instance in enumerate(inputs)
            ]
        )

        if model.world_size > 1 and do_slice:
            prompts = list(islice(inputs, model.rank, len(inputs), model.world_size))
            prompt_entries = list(islice(entries, model.rank, len(entries), model.world_size))
        else:
            prompts = inputs
            prompt_entries = list(entries)

        results = self._generate_with_resume(model, task_name, prompts, prompt_entries)
        if model.world_size > 1:
            all_results = [None for _ in range(model.world_size)]
            dist.all_gather_object(all_results, results)

            all_entries = None
            if not do_slice:
                all_entries = [None for _ in range(model.world_size)]
                dist.all_gather_object(all_entries, prompt_entries)

            # Merge results from all ranks
            length = sum(len(res) for res in all_results if res is not None)
            merged = [None] * length
            for rank, sub_results in enumerate(all_results):
                if sub_results is not None:
                    for i, item in enumerate(sub_results):
                        merged[i * model.world_size + rank] = item
            merged_entries = entries
            if all_entries is not None:
                merged_entries = []
                for rank, rank_entries in enumerate(all_entries):
                    if rank_entries is not None:
                        self.sample_manifest.adopt_entries(rank_entries)
                        for i, entry in enumerate(rank_entries):
                            position = i * model.world_size + rank
                            if position >= len(merged_entries):
                                merged_entries.extend([None] * (position - len(merged_entries) + 1))
                            merged_entries[position] = entry
            self.sample_manifest.mark_generated(merged_entries, merged)
            return merged
        else:
            return results

    def _generate_with_resume(
        self,
        model: LM,
        task_name: str,
        prompts: List[Instance],
        entries: List[SampleEntry],
    ) -> List[str]:
        """Generate over this rank's prompt slice, skipping resume-done units.

        When no manager is attached (the default) this is exactly
        ``model.generate_until(prompts)`` — byte-identical to today (global
        invariant #1). With a manager in ``auto``/``force-fresh`` mode and a
        matching prior run-state, the already-done problems are restored from the
        per-rank manifest and only the remaining problems are regenerated; each
        newly generated output is appended to the manifest as it finishes, then
        restored + new are merged back into ``prompts`` order so grading sees the
        same list it would have seen uninterrupted.

        Unit granularity is per-(problem, repeat) — a wall timeout loses at most
        the in-flight problem, not the whole run.
        """
        manager = getattr(self, "_resume_manager", None)
        # ``_suspend_resume`` is set by the native pass@k batch path
        # (``generate_n_samples_batched``) so its inner ``compute`` calls do NOT
        # double-record at per-(problem,repeat) granularity — pass@k checkpoints at
        # the ``{task, batch_idx}`` unit only (decision #4). When suspended this is a
        # plain generate, exactly as if no manager were attached.
        if manager is None or getattr(self, "_suspend_resume", False):
            outputs = model.generate_until(prompts)
            self.sample_manifest.mark_generated(entries, outputs)
            return outputs

        # Ensure the resume decision (fresh / resume / refuse) is made before we
        # consult done units; ``decide`` is idempotent and is the place a material
        # fingerprint delta refuses loudly (global invariant #3).
        manager.decide()

        restored = manager.restore()  # {unit_key: payload} for completed units
        self.sample_manifest.validate_prior_entries(list(restored.values()), entries)
        keys = [entry.resume_unit(task_name) for entry in entries]

        remaining_instances: List[Instance] = []
        remaining_positions: List[int] = []
        results: List[Optional[str]] = [None] * len(prompts)
        skipped = 0
        from eval.resume import find_restored_payload

        for pos, (inst, unit) in enumerate(zip(prompts, keys)):
            restored_match = find_restored_payload(
                restored,
                [unit, self._unit_key(task_name, inst)],
            )
            if restored_match is not None:
                _, payload = restored_match
                results[pos] = payload["output"]
                skipped += 1
            else:
                remaining_instances.append(inst)
                remaining_positions.append(pos)

        if skipped:
            self.logger.info(
                f"resume[{task_name}] rank={getattr(model, 'rank', 0)}: skipped {skipped} done "
                f"units, generating {len(remaining_instances)} remaining"
            )

        if remaining_instances:
            new_outputs = model.generate_until(remaining_instances)
            self.sample_manifest.validate_output_count(len(remaining_instances), new_outputs)
            for pos, inst, output in zip(remaining_positions, remaining_instances, new_outputs):
                results[pos] = output
                manager.record(
                    keys[pos],
                    {"output": output, "sample": entries[pos].to_dict()},
                )

        manager.finalize()
        self.sample_manifest.mark_generated(entries, results)
        return results

    @abstractmethod
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses from the model for the benchmark tasks."""
        pass

    @abstractmethod
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model's responses according to the benchmark's metrics."""
        pass

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """Run the complete benchmark evaluation pipeline."""
        print(f"Running {self.__class__.__name__} benchmark")
        generation_results = self.generate_responses(model)
        evaluation_results = self.evaluate_responses(generation_results)
        return evaluation_results

    def to_samples(self, generation_result: Dict[str, Any], scored_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reshape generated examples into canonical lm-eval-compatible records.
        lm-eval per-doc sample records (the schema ``save_results_samples`` /
        ``save_results_aggregated`` / ``wandb.log_eval_samples`` consume).

        Default implementation reads the ``{"examples": [...]}`` convention shared
        by the math chat_benchmarks (MATH500/AIME24/AMC23): each ``example`` is the
        source doc enriched in place with ``model_output``/``model_answer`` (single
        sample) or ``model_outputs``/``model_answers`` (native pass@k), with the gold
        in ``example["answer"]``. A benchmark whose ``result`` shape differs should
        override this method.

        Each record mirrors stock ``lm_eval.evaluator.evaluate``'s per-doc dict:
        ``doc_id``, ``doc``, ``target``, ``arguments`` (a list of
        ``[prompt_str, gen_kwargs]`` pairs, so ``save_results_samples``'s
        ``enumerate(sample["arguments"])`` → ``enumerate(arg)`` unpacking works),
        ``resps``/``filtered_resps`` (lists), and the MANDATORY ``doc_hash`` /
        ``prompt_hash`` / ``target_hash`` (``eval_tracker.save_results_aggregated``
        reads all three to build the cumulative task hash).
        """
        import json as _json

        from lm_eval.utils import handle_non_serializable as _hns
        from lm_eval.utils import hash_string

        del scored_result  # Generation owns the per-example data; scoring owns only metrics.
        examples = (generation_result or {}).get("examples", []) or []
        samples: List[Dict[str, Any]] = []
        for doc_id, example in enumerate(examples):
            prompt = self._sample_prompt(example)
            gen_kwargs = self._sample_gen_kwargs(example)
            target = example.get("answer", "")

            if "model_outputs" in example:  # native pass@k: list of completions
                resps = list(example.get("model_outputs", []))
                filtered = list(example.get("model_answers", []))
                extraction_errors = example.get("answer_extraction_errors")
            else:  # single-sample path
                response = next(
                    (
                        example[key]
                        for key in ("model_output", "gpt_completion", "response", "output")
                        if key in example
                    ),
                    "",
                )
                resps = [response]
                filtered = [example.get("model_answer", example.get("generation", response))]
                extraction_error = example.get("answer_extraction_error")
                extraction_errors = [extraction_error] if extraction_error is not None else None

            doc_hash = hash_string(_json.dumps(self._sample_doc(example), indent=2, default=_hns, ensure_ascii=False))
            samples.append(
                {
                    "doc_id": doc_id,
                    "doc": self._sample_doc(example),
                    "target": target,
                    # list-of-pairs: one (prompt, gen_kwargs) "request" per doc.
                    "arguments": [[prompt, gen_kwargs]],
                    "resps": [resps],
                    "filtered_resps": filtered,
                    **({"answer_extraction_errors": extraction_errors} if extraction_errors is not None else {}),
                    "filter": "none",
                    "doc_hash": doc_hash,
                    "prompt_hash": hash_string(prompt),
                    "target_hash": hash_string(str(target)),
                    # The grader's own per-sample scores, in lm-eval's record shape.
                    **sample_metric_fields(example),
                }
            )
        return samples

    def _sample_prompt(self, example: Dict[str, Any]) -> str:
        """Best-effort rendered prompt string for a sample record.

        Default reads a ``problem`` field (the math benches' source field). Override
        for benchmarks whose prompt is built differently.
        """
        return str(example.get("prompt", example.get("problem", example.get("question", example.get("Question", "")))))

    def _sample_doc(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Drop generated fields so ``doc`` and its hash describe source data only."""
        generated_fields = {
            "model_output",
            "model_answer",
            "model_outputs",
            "model_answers",
            "answer_extraction_error",
            "answer_extraction_errors",
            "gpt_completion",
            "generation",
            "response",
            "output",
            "correct",
            "score",
            SAMPLE_METRICS_ANNOTATION,
        }
        return {key: value for key, value in example.items() if key not in generated_fields}

    def _sample_gen_kwargs(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Generation kwargs recorded alongside the prompt in ``arguments``."""
        kwargs: Dict[str, Any] = {}
        max_new = self._evaluation_max_tokens
        if max_new is None:
            max_new = getattr(self, "max_new_tokens", None)
        if max_new is not None:
            kwargs["max_new_tokens"] = max_new
        return kwargs


class TaskManager:
    """
    Enhanced task manager that dynamically loads and manages benchmarks.
    Provides a unified interface for both class-based benchmarks and legacy tasks.
    """

    def __init__(
        self, benchmarks_dir: str = "chat_benchmarks", task_list: Optional[List[str]] = None, **benchmark_kwargs
    ):
        self.logger = logging.getLogger("TaskManager")
        self.tasks: Dict[str, Any] = {}
        self.benchmark_instances: Dict[str, BaseBenchmark] = {}
        self.load_failures: Dict[str, BaseException] = {}
        self.benchmark_kwargs = benchmark_kwargs
        self.task_list = task_list
        self.list_of_tasks_that_require_annotator_model = []
        # Resolved once: every benchmark's generation budget is derived from the
        # same stored prompt lengths (eval/contracts/prompt_lengths.md).
        self.prompt_lengths = load_prompt_lengths()

        # Load benchmarks from directory
        self._load_benchmarks(benchmarks_dir)

    def _load_benchmarks(self, benchmarks_dir: str):
        """Dynamically load benchmarks from the specified directory."""
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), benchmarks_dir)

        # Check if OpenAI API key is available
        has_openai_key = os.getenv("OPENAI_API_KEY") is not None
        if not has_openai_key:
            self.logger.warning("OPENAI_API_KEY not set. Tasks requiring OpenAI will be skipped.")

        # Temporarily set the API key to an empty string to prevent NoneType errors
        if not has_openai_key:
            os.environ["OPENAI_API_KEY"] = ""  # Empty string instead of None

        for item in os.listdir(current_dir):
            # Skip loading if task_list is provided and this item is not in it
            if self.task_list is not None and item not in self.task_list:
                continue

            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            eval_path = os.path.join(item_path, "eval_instruct.py")
            if not os.path.exists(eval_path):
                self.load_failures[item] = FileNotFoundError(f"eval_instruct.py not found in {item}")
                self.logger.warning(f"eval_instruct.py not found in {item}")
                continue

            try:
                # Import the module
                sys.path.insert(0, item_path)
                spec = importlib.util.spec_from_file_location(f"eval.{benchmarks_dir}.{item}.eval_instruct", eval_path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.path.remove(item_path)

                # Find benchmark class
                benchmark_classes = find_custom_benchmark_classes(module, BaseBenchmark)

                if not benchmark_classes:
                    self.load_failures[item] = LookupError(f"No BaseBenchmark subclass found in {item}")
                    self.logger.warning(f"No BaseBenchmark subclass found in {item}")
                    continue

                if len(benchmark_classes) > 1:
                    self.load_failures[item] = LookupError(
                        f"Multiple BaseBenchmark subclasses found in {item}"
                    )
                    self.logger.warning(f"Multiple BaseBenchmark subclasses found in {item}")
                    continue

                benchmark_class = benchmark_classes[0]
                validate_custom_benchmark_class(benchmark_class, BaseBenchmark)

                # Check if this benchmark requires OpenAI as annotator model
                requires_annotator = "annotator_model" in inspect.signature(benchmark_class.__init__).parameters

                # Check if the benchmark explicitly requires OpenAI for annotation
                requires_openai = (
                    hasattr(benchmark_class, "REQUIRES_OPENAI_ANNOTATOR") and benchmark_class.REQUIRES_OPENAI_ANNOTATOR
                )

                if requires_annotator:
                    self.list_of_tasks_that_require_annotator_model.append(item)

                if not has_openai_key and requires_openai:
                    self.logger.warning(
                        f"Not loading {item} benchmark as it requires OpenAI as annotator model but OPENAI_API_KEY is not set"
                    )
                    self.load_failures[item] = RuntimeError("OPENAI_API_KEY is required by this benchmark")
                    continue

                self._register_benchmark(item, benchmark_class)

            except Exception as e:
                self.load_failures[item] = e
                self.logger.error(f"Error loading benchmark from {item}: {str(e)}")
                continue

        # Clean up temporary environment variable if we set it
        if not has_openai_key and "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] == "":
            del os.environ["OPENAI_API_KEY"]

    def _register_benchmark(self, name: str, benchmark_class: Type[BaseBenchmark]):
        """Register a benchmark class and create its instance."""
        try:
            init_params = inspect.signature(benchmark_class.__init__).parameters
            valid_kwargs = {}

            # Only pass kwargs that the benchmark's __init__ accepts
            # Filter out None values to let benchmarks use their default values
            for param_name, param in init_params.items():
                if param_name in self.benchmark_kwargs:
                    value = self.benchmark_kwargs[param_name]
                    # Only pass the argument if it's not None, so benchmarks can use defaults
                    if value is not None:
                        valid_kwargs[param_name] = value
                        self.logger.debug(f"Passing {param_name}={value} to {name} benchmark")

            # Ensure system_instruction is passed if available and not None
            if (
                "system_instruction" in self.benchmark_kwargs
                and self.benchmark_kwargs["system_instruction"] is not None
            ):
                valid_kwargs["system_instruction"] = self.benchmark_kwargs["system_instruction"]

            instance = benchmark_class(**valid_kwargs)
            context_length = self.benchmark_kwargs.get("max_length")
            instance.set_evaluation_limits(
                max_length=context_length,
                max_tokens=resolve_task_max_tokens(
                    name,
                    context_length=context_length,
                    requested_max_tokens=self.benchmark_kwargs.get("max_tokens"),
                    prompt_lengths=self.prompt_lengths,
                ),
                limit=self.benchmark_kwargs.get("limit"),
            )

            self.tasks[name] = benchmark_class
            self.benchmark_instances[name] = instance

            self.logger.debug(f"Successfully registered benchmark: {name}")

        except Exception as e:
            self.load_failures[name] = e
            self.logger.error(f"Error registering benchmark {name}: {str(e)}")

    def get_list_generate_responses(self, task_list: List[str]) -> List[Callable]:
        """Get list of generate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].generate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    def get_list_evaluates(self, task_list: List[str]) -> List[Callable]:
        """Get list of evaluate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].evaluate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    @property
    def available_tasks(self) -> List[str]:
        """Get list of all available tasks."""
        return list(self.tasks.keys())

    def get_benchmark(self, name: str) -> Optional[BaseBenchmark]:
        """Get a benchmark instance by name."""
        return self.benchmark_instances.get(name)

    def is_valid_task(self, task_name: str) -> bool:
        """Check if a task name is valid."""
        return task_name in self.tasks

    def requires_annotator_model(self, task_name: str) -> bool:
        """
        Check if a task requires an annotator model by inspecting its __init__ signature.

        Args:
            task_name: The name of the task to check

        Returns:
            bool: True if the task's __init__ has an annotator_model parameter, False otherwise
        """
        if task_name in self.list_of_tasks_that_require_annotator_model:
            return True
        if task_name not in self.tasks:
            return False

        task_cls = self.tasks[task_name]

        # Get the signature of the task's __init__ method
        init_params = inspect.signature(task_cls.__init__).parameters

        # Check if 'annotator_model' is in the parameters
        return "annotator_model" in init_params


def evaluate(
    lm: LM, task_manager: TaskManager, task_list: List[str], verbosity: str = "INFO", **eval_kwargs
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.

    Args:
        lm: The language model to evaluate
        task_manager: Task manager containing the benchmarks
        task_list: List of task names to evaluate
        verbosity: Logging verbosity level
        **eval_kwargs: Additional kwargs for evaluation

    Returns:
        Dictionary containing evaluation results for each task
    """
    logger = logging.getLogger("evaluate")
    logger.setLevel(getattr(logging, verbosity))

    results = {"results": {}}

    # Validate tasks
    valid_tasks = [t for t in task_list if task_manager.is_valid_task(t)]
    if len(valid_tasks) != len(task_list):
        invalid_tasks = set(task_list) - set(valid_tasks)
        logger.warning(f"Skipping invalid tasks: {invalid_tasks}")

    if not valid_tasks:
        logger.error("No valid tasks to evaluate")
        return results

    # Run evaluations
    for task_name in valid_tasks:
        try:
            benchmark = task_manager.get_benchmark(task_name)
            if benchmark:
                logger.info(f"Evaluating {task_name}")
                results["results"][task_name] = benchmark.run_benchmark(lm)
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {str(e)}")
            results["results"][task_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Initialize task manager
    task_manager = TaskManager()

    # Print available tasks
    print("Available tasks:", task_manager.available_tasks)
