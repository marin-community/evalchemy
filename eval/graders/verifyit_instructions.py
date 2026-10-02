"""Run trusted source instruction predicates through the existing IFEval mode."""

import dataclasses
import importlib
import hashlib
import json
import os
import math
import re
import random
import sys
import tempfile
from pathlib import Path

from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, run
from verifyit.json_objects import unique_object
from verifyit.modes.grade_ifeval import grade_ifeval_candidate
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec, ScriptSpec, render_spec

FAMILIES = {"IFEval": "evaluation", "IFBench": "grader"}
SOURCE_HASHES = {
    "IFEval/evaluation.py": "2a0b4389375f09c3571b874d4a6c7950972384ceed51d15e2d445f59607492c0",
    "IFEval/evaluation_main.py": "7f65f7a92f972a38c7695d4df4178668617586de99b1a5d57b33b8032762c74a",
    "IFEval/instructions.py": "d888e0518118872bf0977ce6bd9b710362b4ae6fb1868c5f6e645a3557d92042",
    "IFEval/instructions_registry.py": "75b84ca1e0f258ffb84896c2999e8dd850ba97757af8ab9e0e72b49fe956974d",
    "IFEval/instructions_util.py": "9c62950b0b27c6c3299122a941b0a17ed3ec85c9bc7ae6432a1359cb6b8e730d",
    "IFBench/grader.py": "1e71c509a1dff80f9c19fc2abde5400f7790e053a0c808f0fadd5b53a2b24415",
    "IFBench/instructions.py": "fe169405312396cb843aa7dda755d160553a4510f47dc186b812dd471da76302",
    "IFBench/instructions_registry.py": "58a4797496ee0d8279d4e3837575123fc3052465f2b7c9a4b3fbc01d88343aed",
    "IFBench/instructions_util.py": "b7d60e07bbb2c56e42ee1a4a1b2f7281ced6c7af3af2719d2dca322afd3c20b8",
}


def _state(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise InvalidTask("Malformed source RNG state")
    version, words, gaussian = value
    if type(version) is not int or version != 3 or not isinstance(words, (list, tuple)) or len(words) != 625:
        raise InvalidTask("Malformed source RNG state")
    if any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words[:-1]):
        raise InvalidTask("Malformed source RNG state")
    if type(words[-1]) is not int or not 0 <= words[-1] <= 624:
        raise InvalidTask("Malformed source RNG position")
    if gaussian is not None and (type(gaussian) not in (int, float) or not math.isfinite(gaussian)):
        raise InvalidTask("Malformed source Gaussian state")
    state = (version, tuple(words), gaussian)
    random.Random().setstate(state)
    return state


def evaluate_accuracy(filename, family):
    if family not in FAMILIES:
        raise InvalidTask("Unknown instruction family")
    with tempfile.TemporaryDirectory(prefix="verifyit-instructions-") as directory:
        root = Path(directory)
        payload = {"family": family, "rows": Path(filename).read_text(), "random_state": random.getstate()}
        (root / "input.json").write_text(json.dumps(payload, allow_nan=False))
        (root / "run.sh").write_text('exec "$@"\n')
        spec = ScriptSpec(
            "run.sh",
            args=(sys.executable, "-m", "eval.graders.verifyit_instructions", str(root / "input.json")),
            timeout=600,
            verdict_file="result.json",
        )
        (root / "verifier.toml").write_text(render_spec(spec))
        verdict = run(root / "verifier.toml", root)
    if verdict.status == Status.INVALID_TASK:
        raise InvalidTask(str(verdict.detail))
    if verdict.status != Status.SCORED:
        raise RuntimeError(f"Instruction verifier did not score: {verdict.status}: {verdict.detail}")
    result = verdict.detail["source_result"]
    rows = [json.loads(line) for line in payload["rows"].splitlines() if line.strip()]
    _validate_result(result, family, rows)
    state = _state(verdict.detail["random_state"])
    random.setstate(state)
    return result


def _metric(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError("Invalid source instruction metric")


def _validate_result(result, family, rows):
    if not isinstance(result, dict):
        raise RuntimeError("Missing source instruction result")
    if family == "IFEval":
        if set(result) != {"prompt-level", "instruction-level", "per_prompt_follow_rate"}:
            raise RuntimeError("Incomplete IFEval source result")
        _metric(result["prompt-level"])
        _metric(result["instruction-level"])
        rates = result["per_prompt_follow_rate"]
        if not isinstance(rates, dict) or set(rates) != {row["prompt"] for row in rows}:
            raise RuntimeError("Incomplete IFEval per-prompt results")
        for value in rates.values():
            if not isinstance(value, dict) or set(value) != {"strict", "loose"}:
                raise RuntimeError("Invalid IFEval protocol results")
            for score in value.values():
                _metric(score)
                if score not in (0, 1):
                    raise RuntimeError("Nonbinary instruction flag")
        if result["prompt-level"] != sum(value["loose"] for value in rates.values()) / len(rows):
            raise RuntimeError("Inconsistent IFEval prompt aggregate")
    else:
        expected = {
            "strict_prompt_accuracy",
            "strict_instruction_accuracy",
            "strict_per_type",
            "loose_prompt_accuracy",
            "loose_instruction_accuracy",
            "loose_per_type",
            "per_prompt_outcomes",
        }
        if set(result) != expected:
            raise RuntimeError("Incomplete IFBench source result")
        categories = {name.split(":")[0] for row in rows for name in row["instruction_id_list"]}
        for protocol in ("strict", "loose"):
            _metric(result[protocol + "_prompt_accuracy"])
            _metric(result[protocol + "_instruction_accuracy"])
            per_type = result[protocol + "_per_type"]
            if not isinstance(per_type, dict) or set(per_type) != categories:
                raise RuntimeError("Incomplete IFBench per-type metrics")
            for metric in per_type.values():
                _metric(metric)
        outcomes = result["per_prompt_outcomes"]
        if not isinstance(outcomes, list) or len(outcomes) != len(rows):
            raise RuntimeError("Incomplete IFBench instruction outcomes")
        for outcome, row in zip(outcomes, rows, strict=True):
            if (
                not isinstance(outcome, dict)
                or set(outcome) != {"prompt", "strict_instruction_pass", "loose_instruction_pass"}
                or outcome["prompt"] != row["prompt"]
            ):
                raise RuntimeError("Invalid IFBench prompt outcome")
            for protocol in ("strict", "loose"):
                flags = outcome[protocol + "_instruction_pass"]
                if (
                    not isinstance(flags, list)
                    or len(flags) != len(row["instruction_id_list"])
                    or any(type(flag) is not bool for flag in flags)
                ):
                    raise RuntimeError("Invalid IFBench instruction flags")

        for protocol in ("strict", "loose"):
            flag_rows = [outcome[protocol + "_instruction_pass"] for outcome in outcomes]
            prompt_accuracy = sum(all(flags) for flags in flag_rows) / len(rows)
            instruction_accuracy = sum(sum(flags) for flags in flag_rows) / sum(len(flags) for flags in flag_rows)
            if (
                result[protocol + "_prompt_accuracy"] != prompt_accuracy
                or result[protocol + "_instruction_accuracy"] != instruction_accuracy
            ):
                raise RuntimeError("Inconsistent IFBench aggregate")
            category_flags = {category: [] for category in categories}
            for row, flags in zip(rows, flag_rows, strict=True):
                for name, flag in zip(row["instruction_id_list"], flags, strict=True):
                    category_flags[name.split(":")[0]].append(flag)
            expected_types = {category: sum(flags) / len(flags) for category, flags in category_flags.items()}
            if result[protocol + "_per_type"] != expected_types:
                raise RuntimeError("Inconsistent IFBench per-type aggregate")


def _arguments(name, arguments):
    for key, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            raise InvalidTask("Empty trusted instruction argument: " + key)
        if isinstance(value, list) and (
            not value or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            raise InvalidTask("Empty or malformed trusted instruction list: " + key)
        if type(value) is bool or isinstance(value, float) and not math.isfinite(value):
            raise InvalidTask("Malformed trusted numeric argument: " + key)
    if name == "keywords:existence":
        for keyword in arguments.get("keywords") or []:
            try:
                if re.search(keyword, ""):
                    raise InvalidTask("Keyword reference matches empty text")
            except re.error as error:
                raise InvalidTask("Malformed keyword regex") from error
    for relation, count in (
        ("relation", "num_words"),
        ("relation", "num_sentences"),
        ("relation", "frequency"),
        ("let_relation", "let_frequency"),
        ("capital_relation", "capital_frequency"),
    ):
        bound = arguments.get(count)
        if arguments.get(relation) == "at least" and type(bound) in (int, float) and bound <= 0:
            raise InvalidTask("Vacuous instruction lower bound")
    if name == "count:word_count_range":
        low, high = arguments.get("min_words"), arguments.get("max_words")
        if type(low) in (int, float) and type(high) in (int, float) and low > high:
            raise InvalidTask("Reversed instruction word range")
    if name == "repeat:repeat_span":
        start, end = arguments.get("n_start"), arguments.get("n_end")
        if type(start) in (int, float) and type(end) in (int, float) and (start < 0 or end < start):
            raise InvalidTask("Invalid trusted repeat span")


def _validate(rows, registry):
    if not rows:
        raise InvalidTask("Instruction batch is empty")
    prompts = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise InvalidTask("Instruction prompt is missing")
        if row["prompt"] in prompts:
            raise InvalidTask("Duplicate instruction prompt would overwrite a response")
        prompts.add(row["prompt"])
        ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")
        if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
            raise InvalidTask("Instruction descriptors are absent or misaligned")
        if any(not isinstance(name, str) or name not in registry for name in ids):
            raise InvalidTask("Unknown source instruction")
        if any(not isinstance(args, dict) for args in kwargs):
            raise InvalidTask("Malformed instruction arguments")
        for name, args in zip(ids, kwargs, strict=True):
            _arguments(name, args)
        if not isinstance(row.get("response"), str):
            raise InvalidTask("Instruction response must be text")
        if "key" not in row:
            raise InvalidTask("Instruction key is missing")


def _install(family, registry, root, observations):
    from verifyit.modes.ifeval import CONSTRAINTS

    for instruction_id, original in tuple(registry.items()):
        name = "evalchemy:" + family + ":" + instruction_id
        if name in CONSTRAINTS:
            raise InvalidTask("Trusted instruction registry collision")

        def build_class(original=original, instruction_id=instruction_id, name=name):
            class VerifiedInstruction(original):
                def check_following(self, response):
                    from eval.graders.verifyit_instruction_data import grade_prepared_instruction

                    prepared = grade_prepared_instruction(family, instruction_id, self, original, response)
                    if prepared is not None:
                        observations.append({"instruction": instruction_id, "verdict": dataclasses.asdict(prepared)})
                        if prepared.status != Status.SCORED:
                            raise RuntimeError("Instruction comparison failed")
                        return prepared.reward == 1
                    error = []

                    def check(candidate, params):
                        try:
                            text = candidate
                            passed = original.check_following(self, text)
                            if type(passed) is not bool:
                                raise RuntimeError("Source instruction did not return a boolean")
                            return passed, "trusted source predicate"
                        except Exception as exception:
                            error.append(f"{type(exception).__name__}: {exception}")
                            return False, "source predicate failed"

                    specification = IfevalSpec((Constraint(name, {}),), empty_output=EmptyOutputPolicy.GRADE)
                    verdict = grade_ifeval_candidate(specification, response, registry={name: check})
                    observations.append({"instruction": instruction_id, "verdict": dataclasses.asdict(verdict)})
                    if verdict.status != Status.SCORED or error:
                        raise RuntimeError("Source instruction execution failed: " + str(error))
                    return verdict.reward == 1

            return VerifiedInstruction

        registry[instruction_id] = build_class()


def main():
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(), object_pairs_hook=unique_object)
    observations = []
    try:
        family = payload["family"]
        if family not in FAMILIES:
            raise InvalidTask("Unknown source family")
        source_root = Path(__file__).resolve().parent.parent / "chat_benchmarks"
        for relative, expected in SOURCE_HASHES.items():
            if (
                relative.startswith(family + "/")
                and hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != expected
            ):
                raise InvalidTask("Trusted instruction source changed: " + relative)
        package = "eval.chat_benchmarks." + family
        module = importlib.import_module(package + "." + FAMILIES[family])
        registry = importlib.import_module(package + ".instructions_registry").INSTRUCTION_DICT
        rows = [
            json.loads(line, object_pairs_hook=unique_object) for line in payload["rows"].splitlines() if line.strip()
        ]
        json.dumps(rows, allow_nan=False)
        _validate(rows, registry)
        import langdetect

        langdetect.DetectorFactory.seed = 0
        random.setstate(_state(payload["random_state"]))
        _install(family, registry, path.parent, observations)
        response_file = path.parent / "responses.jsonl"
        response_file.write_text(payload["rows"])
        try:
            result = module.evaluate_accuracy(str(response_file))
        except (TypeError, ValueError, KeyError, IndexError) as error:
            raise InvalidTask(f"Invalid trusted instruction descriptor: {error}") from error
        components = [
            Reward(item["verdict"]["reward"], Status(item["verdict"]["status"]), item["verdict"]["detail"])
            for item in observations
        ]
        combined = aggregate_rewards(components, expected_total=len(components), policy=Aggregation.ALL)
        verdict = dataclasses.asdict(combined)
        verdict["detail"].update(source_result=result, random_state=random.getstate(), ifeval_calls=observations)
    except InvalidTask as error:
        verdict = {
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"error": str(error), "ifeval_calls": observations},
        }
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "result.json").write_text(json.dumps(verdict, allow_nan=False))


if __name__ == "__main__":
    main()
