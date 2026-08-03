"""GSM8KPerturbed: chat-mode GSM8K under meaning-preserving prompt noise and
irrelevant chat history.

One invocation runs two tasks from checked-in static data (no network access):

    gsm8k-clean      the unperturbed reference (data/gsm8k_clean.jsonl)
    gsm8k-perturbed  one seeded condition per item (data/gsm8k_perturbed.jsonl):
                     noise conditions rewrite only the question text (HELM
                     invariance transforms plus a CMUdict homophone swap);
                     history conditions prepend unrelated GSM8K train-split
                     exchanges as prior chat turns

Reported metrics: clean accuracy, per-condition accuracy and no-answer rate,
and changed-only / spoken-number slices. Data provenance (exact generation
command and parameters) is the meta line of each data file; see
data_prep/generate.py and src/perturbations.py.

Prompting and grading follow lm-eval-harness's gsm8k conventions, zero-shot
through the chat template: the "Question: ... Answer:" prompt and
flexible-extract grading (last number-like token, sanitized numeric match) --
the exact protocol of the prior Marin chat-mode GSM8K measurement
(marin-community/marin#7321).

Design: marin-community/marin#7776 (part of marin-community/marin#7090).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

# lm-eval-harness's gsm8k prompt, sent zero-shot through the chat template --
# the protocol of the prior Marin chat-mode GSM8K measurement
# (marin-community/marin#7321).
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

# lm-eval-harness gsm8k "flexible-extract" convention: last number-like token.
FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLEAN_TASK = "gsm8k-clean"
PERTURBED_TASK = "gsm8k-perturbed"
TASK_FILES = {CLEAN_TASK: "gsm8k_clean.jsonl", PERTURBED_TASK: "gsm8k_perturbed.jsonl"}
NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def load_static_records(filename: str) -> List[Dict[str, Any]]:
    """Read a checked-in data file; line 1 is the provenance meta header."""
    with open(os.path.join(DATA_DIR, filename)) as f:
        header = json.loads(f.readline())
        assert "meta" in header, f"{filename} must start with its provenance meta line"
        return [json.loads(line) for line in f]


def extract_flexible_answer(text: str) -> Optional[str]:
    """lm-eval's flexible-extract: the last number-like token in the output.

    Faithful to the harness convention, a digitless token like ".." still
    counts as an (incorrect) answer rather than no-answer.
    """
    matches = FLEXIBLE_RE.findall(text)
    if not matches:
        return None
    return matches[-1][0] or matches[-1][1]


def sanitize_numeric(answer: str) -> str:
    """Strip currency symbols and thousands separators before comparison."""
    return answer.replace("$", "").replace(",", "").strip()


def numeric_match(model_answer: str, gold: str) -> bool:
    got = NUMBER_RE.search(sanitize_numeric(model_answer))
    want = NUMBER_RE.search(sanitize_numeric(gold))
    return got is not None and want is not None and float(got.group()) == float(want.group())


class GSM8KPerturbedBenchmark(BaseBenchmark):
    def __init__(
        self,
        max_tokens: int = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_tokens
        self.debug = debug

    def _unit_key(self, task_name: str, instance: Instance) -> Dict[str, Any]:
        """Key resume units by the record's globally unique id.

        The base key is (benchmark, instance index); with two tasks both
        numbering instances from zero, a resumed run would replay clean
        outputs as perturbed. Record ids ("gsm8k-test-#N" vs
        "gsm8k-test-#N::<condition>") make collisions impossible.
        """
        return {"task": task_name, "problem_id": str(instance.doc["id"])}

    def _build_instances(self, model: LM, records: List[Dict[str, Any]]) -> List[Instance]:
        instances = []
        for idx, record in enumerate(records):
            messages = []
            for exchange in record.get("history_exchanges") or []:
                messages.append({"role": "user", "content": exchange["question"]})
                messages.append({"role": "assistant", "content": exchange["answer"]})
            prompt = PROMPT_TEMPLATE.replace("{question}", record["question"])
            messages.append({"role": "user", "content": prompt})
            templated = self._prepare_messages(messages, model)
            instances.append(
                Instance(
                    "generate_until",
                    record,
                    (templated, {"do_sample": False, "temperature": 0.0, "max_new_tokens": self.max_new_tokens}),
                    idx,
                )
            )
        return instances

    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for task, filename in TASK_FILES.items():
            records = load_static_records(filename)
            if self.debug:
                records = records[:10]
            for record in records:
                record["task"] = task
            self.logger.info(f"Generating responses for {task} ({len(records)} instances)...")
            outputs = self.compute(model, self._build_instances(model, records))
            if model.rank != 0:
                continue
            for record, output in zip(records, outputs):
                record["output"] = output
            examples.extend(records)
        if model.rank != 0:
            return None
        # The {"examples": [...]} convention feeds BaseBenchmark.to_samples, so
        # per-sample records (--log_samples, wandb) work without an override.
        return {"examples": examples}

    def evaluate_responses(self, results: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if results is None:  # non-primary ranks
            return None
        eval_results: Dict[str, float] = {}
        for record in results["examples"]:
            answer = extract_flexible_answer(record["output"])
            record["model_answer"] = answer
            record["no_answer"] = answer is None
            record["correct"] = answer is not None and numeric_match(answer, record["answer"])
        for task in TASK_FILES:
            records = [r for r in results["examples"] if r["task"] == task]
            if not records:
                continue
            eval_results[task] = _accuracy(records)
            eval_results[f"{task}_no_answer"] = _no_answer_rate(records)
            if task == PERTURBED_TASK:
                eval_results.update(self._per_condition(records, task))
        return eval_results

    @staticmethod
    def _per_condition(records: List[Dict[str, Any]], task: str) -> Dict[str, float]:
        by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            by_condition.setdefault(record["condition"], []).append(record)
        out: Dict[str, float] = {}
        for condition, items in sorted(by_condition.items()):
            prefix = f"{task}:{condition}"
            out[prefix] = _accuracy(items)
            out[f"{prefix}_no_answer"] = _no_answer_rate(items)
            changed = [i for i in items if i.get("changed")]
            if changed and len(changed) != len(items):
                out[f"{prefix}_changed_only"] = _accuracy(changed)
            number_words = [i for i in items if i.get("number_words_affected")]
            if number_words:
                out[f"{prefix}_number_words"] = _accuracy(number_words)
        return out


def _accuracy(records: List[Dict[str, Any]]) -> float:
    return 100.0 * sum(1 for r in records if r["correct"]) / len(records)


def _no_answer_rate(records: List[Dict[str, Any]]) -> float:
    return 100.0 * sum(1 for r in records if r["no_answer"]) / len(records)
