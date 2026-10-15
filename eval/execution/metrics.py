"""Common online reductions for custom evaluation sources."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from eval.execution.engine import ItemResult
from eval.execution.grading import MeanMetrics, TrialMetrics
from eval.contracts.eval_spec import ReducerSpec
from eval.execution.paired import PairedConditions


class SourceMetrics:
    """Reduce scored items, trial cohorts and tagged subgroups in one place."""

    def __init__(self, *, repeated: bool, pass_at_k: Sequence[int] = (), policy: ReducerSpec = ReducerSpec()):
        self.repeated = repeated
        self.policy = policy
        self.means = MeanMetrics()
        self.trials: dict[int, MeanMetrics] = defaultdict(MeanMetrics)
        self.groups: dict[tuple[str, str], MeanMetrics] = defaultdict(MeanMetrics)
        self.group_trials = defaultdict(lambda: defaultdict(MeanMetrics))
        self.examples: set[str] = set()
        self.squared_accuracy = 0.0
        if pass_at_k and policy.trial_metric is None:
            raise ValueError("pass@k requires a binary trial metric")
        self.pass_at_k = TrialMetrics(pass_at_k, policy.trial_metric) if pass_at_k else None
        self.paired = PairedConditions() if policy.name == "paired_conditions" else None
        self.annotations = [] if policy.name == "alpaca" else None

    def add(self, result: ItemResult) -> None:
        self.repeated |= result.spec.trial is not None
        if self.paired is not None:
            self.paired.add(result)
        if self.annotations is not None:
            self.annotations.append(result.grade.audit["alpaca_annotation"])
        self.means.add(result.grade.metrics, result.grade.weights)
        self.examples.add(result.record["sample_id"])
        self.trials[result.spec.trial or 0].add(result.grade.metrics, result.grade.weights)
        self.squared_accuracy += float(result.grade.metrics.get("accuracy", 0)) ** 2
        for tag, values in result.spec.tags.items():
            for value in (values,) if isinstance(values, str) else values:
                self.groups[tag, value].add(result.grade.metrics, result.grade.weights)
                self.group_trials[tag, value][result.spec.trial or 0].add(result.grade.metrics, result.grade.weights)
        if self.pass_at_k is not None:
            self.pass_at_k.add(result)

    def finish(self, task: str) -> dict:
        metrics = self.means.finish()
        count = len(self.examples)
        if self.pass_at_k is not None:
            metrics = self.pass_at_k.finish()
        elif "accuracy" in metrics:
            accuracy = metrics.pop("accuracy")
            accuracy_name = "accuracy_avg" if self.repeated and self.policy.accuracy == "accuracy" else self.policy.accuracy
            stderr_name = "accuracy_std_err" if self.repeated and self.policy.stderr == "accuracy_stderr" else self.policy.stderr
            metrics[accuracy_name] = accuracy
            if self.repeated:
                accuracies = [self.trials[trial].finish()["accuracy"] for trial in sorted(self.trials)]
                metrics.update(
                    num_total=count, num_repeat=len(accuracies), solved_avg=accuracy * count,
                    run_stats=[
                        {"repetition": trial + 1, "num_total": count, "num_solved": value * count, "accuracy": value}
                        for trial, value in enumerate(accuracies)
                    ],
                )
                denominator = len(accuracies) * (len(accuracies) - self.policy.repeat_ddof)
                metrics[stderr_name] = math.sqrt(sum((value - accuracy) ** 2 for value in accuracies) / denominator) if denominator > 0 else 0.0
            else:
                variance = max(0, self.squared_accuracy - count * accuracy ** 2) / (count - 1) if count > 1 else 0
                metrics.update(num_total=count, num_solved=accuracy * count)
                metrics[stderr_name] = math.sqrt(variance / count)
        for (tag, group), accumulator in self.groups.items():
            for metric, value in accumulator.finish().items():
                if tag in self.policy.groups:
                    name = self.policy.groups[tag].format(group=group, metric=self.policy.accuracy if metric == "accuracy" else metric)
                else:
                    name = f"{metric}_{group}{self.policy.group_suffix}" if metric == "accuracy" else f"{tag}:{group}/{metric}"
                metrics[name] = value
                if metric == "accuracy" and self.repeated and self.policy.group_suffix == "_avg":
                    trials = [trial.finish()[metric] for trial in self.group_trials[tag, group].values()]
                    denominator = len(trials) * (len(trials) - self.policy.repeat_ddof)
                    metrics[f"accuracy_{group}_std_err"] = math.sqrt(sum((score - value) ** 2 for score in trials) / denominator) if denominator > 0 else 0.0
            if "accuracy" in accumulator.counts:
                metrics[f"count_{group}"] = accumulator.counts["accuracy"]
        if self.policy.name == "classification":
            correct = self.means.sums["accuracy"]
            incorrect = self.means.sums["incorrect"]
            not_attempted = self.means.sums["not_attempted"]
            judged = correct + incorrect + not_attempted
            accuracy = correct / judged if judged else 0.0
            attempted = correct / (correct + incorrect) if correct + incorrect else 0.0
            metrics.update(
                accuracy=accuracy, correct_rate=accuracy, incorrect_rate=incorrect / judged if judged else 0.0,
                not_attempted_rate=not_attempted / judged if judged else 0.0,
                accuracy_given_attempted=attempted, f1=2 * accuracy * attempted / (accuracy + attempted) if accuracy + attempted else 0.0,
                num_judged=judged, num_correct=correct, num_incorrect=incorrect, num_not_attempted=not_attempted,
                num_judge_failed=self.means.sums["judge_failed"], judge_coverage=judged / count,
            )
        if self.policy.name == "instructions":
            for name in list(metrics):
                if ":" in name:
                    group, member = name.split(":", 1)
                    metrics.setdefault(group, {})[member] = metrics.pop(name)
        if self.policy.name == "weighted_macro":
            categories = {value: accumulator.finish()["adjusted_score"] for (tag, value), accumulator in self.groups.items()
                          if value in self.policy.group_weights}
            macro = sum(value * self.policy.group_weights[name] for name, value in categories.items()) / sum(self.policy.group_weights.values())
            metrics.update(task_macro_score=macro, adjusted_task_macro_score=macro, task_categorized_scores=categories)
        if self.paired is not None:
            metrics = self.paired.finish()
        if self.annotations is not None:
            from alpaca_eval.metrics import get_length_controlled_winrate

            metrics = get_length_controlled_winrate(self.annotations, save_weights_dir=None)
        return {"results": {task: metrics}}
