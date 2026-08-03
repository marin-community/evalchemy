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

Prompting and grading follow ZeroEval's open-ended QA protocol (JSON
reasoning/answer output, sanitized numeric match), reimplemented here so the
benchmark is self-contained.

Design: marin-community/marin#7776 (part of marin-community/marin#7090).
"""

import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

# ZeroEval's open-ended QA template (github.com/WildEval/ZeroEval, Apache-2.0).
PROMPT_TEMPLATE = """
## Question:

{question}


## Instruction

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format.
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""

# lm-eval-harness gsm8k "flexible-extract" convention: last number-like token.
FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TASK_FILES = {"gsm8k-clean": "gsm8k_clean.jsonl", "gsm8k-perturbed": "gsm8k_perturbed.jsonl"}
NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def load_static_records(filename: str) -> List[Dict[str, Any]]:
    """Read a checked-in data file, skipping its provenance meta line."""
    records = []
    with open(os.path.join(DATA_DIR, filename)) as f:
        for line in f:
            record = json.loads(line)
            if "meta" not in record:
                records.append(record)
    return records


def extract_json_answer(text: str) -> Optional[str]:
    """Return the `answer` field of the first or last complete JSON object."""
    for candidate in (_first_json(text), _last_json(text)):
        if candidate is not None and "answer" in candidate:
            return str(candidate["answer"])
    return None


def _first_json(s: str) -> Optional[dict]:
    return _scan_json(s, take_last=False)


def _last_json(s: str) -> Optional[dict]:
    return _scan_json(s, take_last=True)


def _scan_json(s: str, take_last: bool) -> Optional[dict]:
    stack: List[int] = []
    start = None
    found = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if start is None:
                start = i
        elif char == "}" and stack:
            stack.pop()
            if not stack:
                snippet = s[start : i + 1]
                start = None
                try:
                    parsed = json.loads(snippet.replace("\n", ""))
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    if not take_last:
                        return parsed
                    found = parsed
    return found


def extract_flexible_answer(text: str) -> Optional[str]:
    """lm-eval's flexible-extract: the last number-like token in the output.

    Reported alongside the JSON metric as a format-free secondary: divergence
    between the two localizes format breakdown vs. wrong math.
    """
    matches = FLEXIBLE_RE.findall(text)
    if not matches:
        return None
    return matches[-1][0] or matches[-1][1]


def sanitize_numeric(answer: str) -> str:
    """ZeroEval's math sanitization: strip $/commas, evaluate plain fractions."""
    answer = answer.replace("$", "").replace(",", "").strip()
    if "/" in answer:
        try:
            answer = str(float(eval(answer)))  # noqa: S307 - "a/b" strings from our own regex-checked answers
        except Exception:
            pass
    return answer


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

    def _build_instances(self, model: LM, records: List[Dict[str, Any]], idx_offset: int) -> List[Instance]:
        # Instance indices are offset per task: the resume store keys units by
        # (benchmark, problem_idx), so the clean and perturbed tasks must not
        # reuse indices or a resumed run replays clean outputs as perturbed.
        instances = []
        for idx, record in enumerate(records, start=idx_offset):
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

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        temp_dir_obj = tempfile.TemporaryDirectory()
        results: Dict[str, Any] = {"temp_dir_obj": temp_dir_obj}
        for task_index, (task, filename) in enumerate(TASK_FILES.items()):
            records = load_static_records(filename)
            if self.debug:
                records = records[:10]
            self.logger.info(f"Generating responses for {task} ({len(records)} instances)...")
            outputs = self.compute(model, self._build_instances(model, records, idx_offset=task_index * 100_000))
            if model.rank != 0:
                continue
            for record, output in zip(records, outputs):
                record["output"] = output
            output_path = os.path.join(temp_dir_obj.name, f"{task}.json")
            with open(output_path, "w") as f:
                json.dump(records, f)
            results[task] = output_path
        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        temp_dir_obj = results.pop("temp_dir_obj")
        eval_results: Dict[str, float] = {}
        for task, filepath in results.items():
            with open(filepath) as f:
                records = json.load(f)
            for record in records:
                answer = extract_json_answer(record["output"])
                record["no_answer"] = answer is None
                record["correct"] = answer is not None and numeric_match(answer, record["answer"])
                flexible = extract_flexible_answer(record["output"])
                record["flexible_correct"] = flexible is not None and numeric_match(flexible, record["answer"])
            eval_results[task] = _accuracy(records)
            eval_results[f"{task}_no_answer"] = _no_answer_rate(records)
            eval_results[f"{task}_flexible"] = _flexible_accuracy(records)
            if task == "gsm8k-perturbed":
                eval_results.update(self._per_condition(records))
        temp_dir_obj.cleanup()
        return eval_results

    @staticmethod
    def _per_condition(records: List[Dict[str, Any]]) -> Dict[str, float]:
        by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            by_condition.setdefault(record["condition"], []).append(record)
        out: Dict[str, float] = {}
        for condition, items in sorted(by_condition.items()):
            prefix = f"gsm8k-perturbed:{condition}"
            out[prefix] = _accuracy(items)
            out[f"{prefix}_no_answer"] = _no_answer_rate(items)
            out[f"{prefix}_flexible"] = _flexible_accuracy(items)
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


def _flexible_accuracy(records: List[Dict[str, Any]]) -> float:
    return 100.0 * sum(1 for r in records if r["flexible_correct"]) / len(records)
