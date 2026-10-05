"""Prepare LiveBench reasoning answers for existing core comparisons."""

import itertools
import re

from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.spec import ExactSpec

NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def _literal(reference, answer, *, substring=False):
    return grade_exact_candidate(
        ExactSpec(
            expected=(reference,),
            ignore_case=False,
            ignore_whitespace=False,
            strip_outer_whitespace=False,
            substring=substring,
        ),
        answer,
    )


def _boxed(response):
    from livebench.process_results.util import last_boxed_only_string, remove_boxed

    value = last_boxed_only_string(response.replace("\\\\fbox{", "\\\\boxed{"))
    return remove_boxed(value) if value else None


def zebra(question, response):
    reference = question["ground_truth"]
    if question["livebench_release_date"] < "2024-11-25":
        bold = re.findall(r"\*\*\*(\w+)\*\*\*", response)
        words = bold or re.findall(r"\b\w+\b", response)
        answer = words[-1].lower() if words else ""
        numbers = {str(i): word for i, word in enumerate(NUMBER_WORDS) if 1 <= i <= 9}
        candidates = [answer, numbers.get(answer, ""), answer + " movies"]
        return aggregate_rewards(
            [_literal(reference.lower(), value) for value in candidates], expected_total=3, policy=Aggregation.MAX
        )
    references = [word.strip().lower().replace("-", " ") for word in reference.split(",")]
    if any(not word for word in references):
        raise InvalidTask("Zebra references must contain nonempty answers")
    matches = re.findall(r"<solution>(.*?)</solution>", response) or re.findall(
        r"</solution>(.*?)</solution>", response
    )
    if not matches:
        boxed = _boxed(response)
        if boxed is not None:
            matches = [boxed.replace("\\text{", "").replace("}", "").replace("\\", "")]
    if not matches:
        line = response.strip().split("\n")[-1]
        if line.count(",") == len(references) - 1:
            matches = [line]
    answers = [word for match in matches for word in match.split(",")]
    if len(matches) > 1:
        answers = answers[-len(references) :]
    grades = [
        _literal(
            word,
            answers[i].strip().lower().replace("-", " ").replace("position", "") if i < len(answers) else "",
            substring=True,
        )
        for i, word in enumerate(references)
    ]
    components = [
        aggregate_rewards(grades, expected_total=len(references), policy=policy)
        for policy in (Aggregation.ALL, Aggregation.MEAN)
    ]
    return aggregate_rewards(components, expected_total=2, policy=Aggregation.MEAN)


def web_of_lies(question, response):
    reference = question["ground_truth"].lower()
    bold = re.findall(r"\*\*(.*?)\*\*", response)
    words = [
        word.lower().strip().replace(",", "").replace(".", "")[: max(len(word), 3)]
        for match in bold
        for word in match.split()
    ]
    words = [word for word in words if word in ("yes", "no")][-3:]
    answer = ", ".join(words) if words else None
    if answer is None:
        prepared = response.replace("\\\\boxed{\\\\textbf{", "\\\\boxed{").replace("\\textbf{", "\\boxed{")
        answer = _boxed(prepared)
    if answer is None:
        alternatives = [
            (response.lower().find(", ".join(values)), values) for values in itertools.product(("yes", "no"), repeat=3)
        ]
        index, values = max(alternatives, key=lambda pair: pair[0])
        answer = ", ".join(values) if index >= 0 else ""
    schema = {
        "type": "object",
        "required": ["answer", "tokens"],
        "anyOf": [
            {"properties": {"answer": {"const": reference}}},
            {
                "properties": {
                    "answer": {"type": "string", "pattern": re.escape(reference)},
                    "tokens": {"type": "array", "minItems": 3, "maxItems": 3},
                }
            },
        ],
    }
    return grade_json_schema_candidate(schema, {"answer": answer, "tokens": re.findall("yes|no", answer)})


def spatial(question, response):
    reference = question["ground_truth"]
    normalized = reference.strip().lower()
    numbers = {word: str(i) for i, word in enumerate(NUMBER_WORDS)}
    bold = re.findall(r"\*\*([^\*]+)\*\*", response)[-3:]
    grades = []
    for raw in bold:
        word = raw.strip().lower()
        grades.append(_literal(normalized, word))
        grades.append(_literal(normalized, numbers.get(word, "")))
        if normalized in ("tetrahedra", "tetrahedron", "triangle", "square"):
            grades.append(
                grade_json_schema_candidate(
                    {"type": "string", "pattern": re.escape(normalized), "maxLength": 2 * len(normalized) + 4}, word
                )
            )
    boxed = _boxed(response)
    prepared = boxed.replace("\\textbf{", "").replace("\\mathbf{", "").replace("}", "") if boxed else ""
    grades.append(_literal(reference, prepared))
    return aggregate_rewards(grades, expected_total=len(grades), policy=Aggregation.MAX)
