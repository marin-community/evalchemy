"""LiveBench opt-in grading, preserving its client aggregation and loaders."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
import uuid

from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, scored
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.modes.grade_script import grade_script_callable
from verifyit.spec import ExactSpec


def validate_sources():
    root = Path(__file__).resolve().parents[2]
    hashes = json.loads(Path(__file__).with_name("livebench_runtime").joinpath("source-hashes.json").read_text())
    for relative, expected in hashes.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise InvalidTask(f"LiveBench source contract changed: {relative}")


def grade_retained(question, response):
    from eval.graders.verifyit_instructions import _state

    name = "verifyit-livebench-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="verifyit-livebench-") as directory:
        root = Path(directory)
        payload = {
            "question": question,
            "response": response,
            "random_state": random.getstate(),
            "container": name,
        }
        try:
            verdict = grade_script_callable(_grade_payload, payload, root, timeout=120)
        finally:
            task = question["subtask"] if "subtask" in question else question["task"]
            if task in ("coding_completion", "LCB_generation") or "amps_hard" in task:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
    if verdict.status == Status.INVALID_TASK:
        raise InvalidTask(str(verdict.detail))
    if verdict.status != Status.SCORED:
        raise RuntimeError(f"LiveBench retained verifier failed: {verdict.status}: {verdict.detail}")
    if verdict.detail.get("source_score") != verdict.reward:
        raise RuntimeError("LiveBench returned inconsistent source score")
    if question.get("category") == "instruction_following":
        flags = verdict.detail.get("instruction_flags")
        if (
            not isinstance(flags, list)
            or len(flags) != len(question["instruction_id_list"])
            or not flags
            or any(type(flag) is not bool for flag in flags)
        ):
            raise RuntimeError("LiveBench returned inconsistent instruction outcomes")
    random.setstate(_state(verdict.detail["random_state"]))
    return verdict


def runtime_image():
    return os.environ.get("VERIFYIT_LIVEBENCH_IMAGE", "verifyit-evalchemy-livebench:source-v1")


def runtime_score(payload, path):
    root = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name("verifyit_livebench_worker.py")
    image = runtime_image()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            payload["container"],
            "-v",
            f"{root}:/source:ro",
            "-v",
            f"{path.parent}:/work",
            "-e",
            "PYTHONPATH=/source:/source/eval/chat_benchmarks/LiveBench",
            image,
            "python",
            "/source/" + str(worker.relative_to(root)),
            "/work/input.json",
            "/work/runtime-result.json",
        ],
        check=True,
        timeout=110,
    )
    result = json.loads((path.parent / "runtime-result.json").read_text())
    if result.get("status") == "invalid_task":
        raise InvalidTask(result.get("error", "Invalid LiveBench runtime task"))
    score = result.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("Invalid LiveBench runtime result")
    return score


def instruction_score(question, response, root):
    import langdetect
    from eval.graders.verifyit_instructions import _install, _validate
    from livebench.if_runner.instruction_following_eval import evaluation_main

    registry = evaluation_main.instructions_registry.INSTRUCTION_DICT
    row = {
        "key": question["question_id"],
        "prompt": question["turns"][0],
        "instruction_id_list": question.get("instruction_id_list"),
        "kwargs": question.get("kwargs"),
        "response": response,
    }
    _validate([row], registry)
    langdetect.DetectorFactory.seed = 0
    for identifier in ("language:response_language", "change_case:english_capital", "change_case:english_lowercase"):
        original = registry[identifier]

        def corrected(original=original, identifier=identifier):
            class LanguageInstruction(original):
                def check_following(self, value):
                    try:
                        if identifier == "change_case:english_capital":
                            return value.isupper() and langdetect.detect(value) == "en"
                        if identifier == "change_case:english_lowercase":
                            return value.islower() and langdetect.detect(value) == "en"
                        return langdetect.detect(value) == self._language
                    except langdetect.LangDetectException:
                        return False

            return LanguageInstruction

        registry[identifier] = corrected()
    observations = []
    _install("LiveBench", registry, root, observations)
    inputs = evaluation_main.read_prompt_list([question])
    result = evaluation_main.test_instruction_following_strict(inputs[0], {row["prompt"]: response})
    flags = result.follow_instruction_list
    if not flags or any(type(flag) is not bool for flag in flags):
        raise RuntimeError("Invalid LiveBench instruction outcomes")
    components = [
        Reward(item["verdict"]["reward"], Status(item["verdict"]["status"]), item["verdict"]["detail"])
        for item in observations
    ]
    all_pass = aggregate_rewards(components, expected_total=len(flags), policy=Aggregation.ALL)
    partial = aggregate_rewards(components, expected_total=len(flags), policy=Aggregation.MEAN)
    verdict = aggregate_rewards([all_pass, partial], expected_total=2, policy=Aggregation.MEAN)
    if verdict.status is not Status.SCORED:
        raise RuntimeError(f"Instruction grading failed: {verdict.detail}")
    return verdict.reward, {"instruction_flags": flags, "instruction_verdicts": observations}


def retained_score(question, response):
    from livebench import gen_ground_truth_judgment as source

    task = question["subtask"] if "subtask" in question else question["task"]
    reference = question.get("ground_truth")
    if not isinstance(reference, str) or not reference.strip():
        raise InvalidTask("LiveBench requires a nonempty trusted reference")
    parts = task.split("_")
    if parts[0] in ("amc", "smc") or (len(parts) > 1 and parts[1] == "amc"):
        if len(reference) != 1 or reference not in "ABCDE":
            raise InvalidTask("AMC/SMC reference must be one of A through E")
        from livebench.process_results.math.math_competitions.utils import extract_answer
        from livebench.process_results.util import last_boxed_only_string, remove_boxed

        boxed = last_boxed_only_string(response.replace("\\\\fbox{", "\\\\boxed{"))
        prepared = (
            remove_boxed(boxed).replace("\\text{", "").replace("}", "").replace("\\", "").lower() if boxed else ""
        )
        value = extract_answer(question["turns"][0], reference)
        alternatives = [
            grade_exact_candidate(ExactSpec(expected=(reference * 4,), ignore_case=False, substring=True), response),
            grade_exact_candidate(
                ExactSpec(
                    expected=(reference.lower(),),
                    ignore_case=False,
                    ignore_whitespace=False,
                    strip_outer_whitespace=False,
                ),
                prepared,
            ),
            grade_exact_candidate(
                ExactSpec(
                    expected=(value,),
                    ignore_case=False,
                    ignore_whitespace=False,
                    strip_outer_whitespace=False,
                    substring=True,
                ),
                response[-20 - len(value) :],
            ),
        ]
        return aggregate_rewards(alternatives, expected_total=3, policy=Aggregation.MAX).reward
    if parts[0] == "aime":
        return grade_exact_candidate(
            ExactSpec(
                expected=(reference,),
                ignore_case=False,
                ignore_whitespace=False,
                strip_outer_whitespace=False,
                substring=True,
            ),
            response[-50:],
        ).reward
    if task == "typos":
        from livebench.process_results.writing.typos.utils import extract_answer

        prepared = extract_answer(" ".join(filter(None, response.split("\n"))))
        return grade_exact_candidate(
            ExactSpec(
                expected=(reference,),
                ignore_case=False,
                ignore_whitespace=False,
                strip_outer_whitespace=False,
                substring=True,
            ),
            prepared,
        ).reward
    if parts[0] in ("imo", "usamo"):
        try:
            [int(number) for number in reference.split(",")]
        except ValueError as error:
            raise InvalidTask("Proof rearrangement reference must be a nonempty integer sequence") from error
        return source.proof_rearrangement_process_results(reference, response, edit_distance=True, debug=False)
    if task == "tablereformat":
        from livebench.process_results.data_analysis.tablereformat.utils import read_df_func

        try:
            output_format = (
                question["turns"][0]
                .split("Please convert the Input Table from ")[1]
                .split("format to ")[1]
                .split(" format")[0]
                .lower()
            )
            table = read_df_func(output_format, reference)
            if table is None or table.empty:
                raise ValueError("Empty trusted table")
        except (ValueError, TypeError, IndexError) as error:
            raise InvalidTask("Invalid trusted table format or reference") from error
        return source.table_process_results(question["turns"][0], reference, response, False)
    if task == "tablejoin":
        try:
            mapping = ast.literal_eval(reference)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Empty join reference")
        except (ValueError, TypeError, SyntaxError) as error:
            raise InvalidTask("Table join reference must be a nonempty mapping") from error
        return source.joinmap_process_results(question["turns"][0], reference, response, False)
    if task in ("zebra_puzzle", "web_of_lies_v2", "spatial"):
        from eval.graders import verifyit_livebench_reasoning as reasoning

        prepare = {"zebra_puzzle": reasoning.zebra, "web_of_lies_v2": reasoning.web_of_lies, "spatial": reasoning.spatial}[task]
        return prepare(question, response).reward
    if task == "connections":
        if any(not word.strip() for word in reference.split(",")):
            raise InvalidTask("Connections reference contains an empty word")
        return grade_connections(question, response).reward
    callbacks = {
        "house_traversal": source.house_traversal_process_results,
        "plot_unscrambling": source.plot_unscrambling_process_results,
    }
    if task not in callbacks:
        raise InvalidTask(f"LiveBench has no enabled grading route for {task}")
    if task == "plot_unscrambling" and not any(sentence.strip() for sentence in reference.split(".")):
        raise InvalidTask("Plot reference has no sentences")
    return callbacks[task](reference, response, False)


def grade_connections(question, response):
    from livebench.process_results.writing.connections.utils import group_words
    from livebench.process_results.util import last_boxed_only_string, remove_boxed

    reference = question["ground_truth"].split(",")
    words = [word.strip().lower() for word in reference]
    if len(set(words)) != len(words):
        raise InvalidTask("Connections trusted groups must contain distinct words")
    groups = [sorted(group) for group in group_words(reference)]
    old = question["livebench_release_date"] < "2024-11-25"
    if old:
        solutions = re.findall(r"\*\*(.*?)\*\*", response.replace("\n", ""))
    else:
        solutions = []
        for tag in ("<solution>", "</solution>"):
            for text in (response, response.replace("\n", "")):
                solutions = re.findall(tag + r"(.*?)</solution>", text)
                if solutions:
                    break
            if solutions:
                break
        if not solutions and "\\boxed" in response:
            boxed = last_boxed_only_string(response)
            parsed = remove_boxed(boxed) if boxed else ""
            solutions = [parsed.replace("\\text{", "").replace("}", "").replace("\\", "")]
        solutions = [match.replace("\n", "") for match in solutions]
        if len(solutions) > 1:
            solutions = [",".join(",".join(solutions).split(",")[-len(reference) :])]
    alternatives = []
    for solution in solutions:
        candidate = [sorted(group) for group in group_words(solution.split(","))]
        components = []
        for group in groups:
            member = (
                {"type": "array", "allOf": [{"contains": {"const": word}} for word in group]}
                if old
                else {"const": group}
            )
            components.append(grade_json_schema_candidate({"type": "array", "contains": member}, candidate))
        alternatives.append(aggregate_rewards(components, expected_total=len(groups), policy=Aggregation.MEAN))
    return aggregate_rewards(alternatives, expected_total=max(1, len(solutions)), policy=Aggregation.MAX)


def grade_cta(question, response):
    from livebench.process_results.data_analysis.cta.utils import clean_text
    from livebench.process_results.util import last_boxed_only_string, remove_boxed

    reference = question.get("ground_truth")
    if not isinstance(reference, str) or not clean_text(reference):
        raise InvalidTask("LiveBench CTA requires a nonempty normalized reference")
    if not isinstance(response, str):
        raise InvalidTask("LiveBench response must be text")
    parsed = response
    if "\\boxed{" in parsed:
        boxed = last_boxed_only_string(parsed)
        parsed = remove_boxed(boxed) if boxed is not None else ""
        parsed = (parsed or "").replace("\\text{", "").replace("}", "").replace("\\", "")
    expected = clean_text(reference)
    return grade_exact_candidate(
        ExactSpec(expected=(expected,), ignore_case=False, ignore_whitespace=False),
        clean_text(parsed)[-len(expected) :],
    )


class JudgmentBatch:
    """Write judgments from current responses, removing artifacts on batch failure."""

    def __init__(self, responses):
        validate_sources()
        if not isinstance(responses, list) or not responses:
            raise InvalidTask("LiveBench requires current response records")
        self.responses = {}
        self.outputs = set()
        for response in responses:
            if not isinstance(response, dict):
                raise InvalidTask("LiveBench response record must be an object")
            key = (response.get("model_id"), response.get("question_id"))
            if not isinstance(key[0], str) or not key[0] or key[1] is None or key in self.responses:
                raise InvalidTask("LiveBench responses need unique model/question identities")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise InvalidTask("LiveBench requires a first response choice")
            turns = choices[0].get("turns")
            if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) for turn in turns):
                raise InvalidTask("LiveBench response turns must contain text")
            self.responses[key] = response
        if len({model for model, _ in self.responses}) != 1:
            raise InvalidTask("LiveBench evaluation accepts exactly one model")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            for output in self.outputs:
                output.unlink(missing_ok=True)

    def __call__(self, *, questions, output_file, model_list, **kwargs):
        output = Path(output_file)
        self.outputs.add(output)
        output.unlink(missing_ok=True)
        if not questions or len(model_list) != 1:
            raise InvalidTask("LiveBench grading requires questions and one model")
        model = model_list[0]
        judgments = []
        if questions[0].get("category") == "instruction_following":
            prompts = [question["turns"][0] for question in questions]
            if len(set(prompts)) != len(prompts):
                raise InvalidTask("Duplicate instruction prompts would overwrite source responses")
        for question in questions:
            if (
                not isinstance(question, dict)
                or not isinstance(question.get("turns"), list)
                or not question["turns"]
                or not isinstance(question["turns"][0], str)
                or not question["turns"][0].strip()
            ):
                raise InvalidTask("LiveBench trusted question must contain a prompt")
            response = self.responses.get((model, question.get("question_id")))
            if response is None:
                raise InvalidTask("LiveBench current response is missing")
            task = question["subtask"] if "subtask" in question else question.get("task")
            if not isinstance(task, str) or not task:
                raise InvalidTask("LiveBench requires a nonempty task or subtask")
            grader = grade_cta if task == "cta" else grade_retained
            turn = 0 if question.get("category") == "instruction_following" else -1
            verdict = grader(question, response["choices"][0]["turns"][turn])
            if verdict.status != Status.SCORED:
                raise RuntimeError(f"LiveBench verifier failed: {verdict.status}")
            judgments.append(
                {
                    "question_id": question["question_id"],
                    "task": question["task"],
                    "category": question["category"],
                    "model": model,
                    "turn": 1,
                    "score": verdict.reward,
                    "tstamp": time.time(),
                }
            )
            if "subtask" in question:
                judgments[-1]["subtask"] = question["subtask"]
        output.parent.mkdir(parents=True, exist_ok=True)
        judgments.sort(key=lambda row: (row["question_id"], row["model"]))
        output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in judgments))


def _grade_payload(payload, root):
    from eval.graders.verifyit_instructions import _state

    validate_sources()
    random.setstate(_state(payload["random_state"]))
    detail = {}
    task = payload["question"].get("subtask", payload["question"]["task"])
    if task in ("coding_completion", "LCB_generation"):
        from eval.graders.verifyit_livebench_worker import grade_coding

        verdict = grade_coding(payload["question"], payload["response"], payload["container"], image=runtime_image())
        verdict.detail.update(source_score=verdict.reward, random_state=random.getstate())
        return verdict
    elif "amps_hard" in task:
        path = root / "input.json"
        path.write_text(json.dumps(payload, allow_nan=False))
        score = runtime_score(payload, path)
    elif payload["question"].get("category") == "instruction_following":
        score, detail = instruction_score(payload["question"], payload["response"], root)
    else:
        try:
            score = retained_score(payload["question"], payload["response"])
        except InvalidTask:
            raise
        except (AssertionError, IndexError, AttributeError, ValueError, TypeError) as error:
            score = 0
            detail["candidate_parse_error"] = f"{type(error).__name__}: {error}"
    return scored(score, source_score=score, random_state=random.getstate(), **detail)
