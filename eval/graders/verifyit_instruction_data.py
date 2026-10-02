"""Prepare source instruction observations for existing core comparisons."""

import json
import re

from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards
from verifyit.json_objects import unique_object
from verifyit.modes.grade_ifeval import grade_ifeval_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec

MAPPED_IDS = {
    "keywords:frequency",
    "keywords:letter_frequency",
    "detectable_content:number_placeholders",
    "detectable_format:number_highlighted_sections",
    "change_case:capital_word_frequency",
    "punctuation:no_comma",
    "language:response_language",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "keywords:existence",
    "keywords:forbidden_words",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists",
    "detectable_format:multiple_sections",
    "detectable_format:constrained_response",
    "detectable_format:title",
    "detectable_format:json_format",
    "detectable_content:postscript",
    "startend:quotation",
    "startend:end_checker",
    "combination:repeat_prompt",
    "combination:two_responses",
}


def _count_schema(expected, relation="exactly"):
    if type(expected) is not int or expected < 0:
        raise InvalidTask("Instruction count must be a nonnegative integer")
    bounds = {"exactly": "const", "less than": "exclusiveMaximum", "at least": "minimum"}
    if relation not in bounds:
        raise InvalidTask("Unsupported instruction count relation")
    return {"type": "integer", bounds[relation]: expected}


def grade_prepared_instruction(family, identifier, instruction, original, text):
    """Use source builders/tokenizers only; core owns every acceptance decision."""
    if family not in {"LiveBench", "IFEval"} or identifier not in MAPPED_IDS:
        return None
    args = instruction.get_instruction_args() or {}
    if identifier == "combination:two_responses":
        return grade_ifeval_candidate(
            IfevalSpec((Constraint(identifier, {}),), empty_output=EmptyOutputPolicy.GRADE), text
        )
    if identifier in {"language:response_language", "change_case:english_capital", "change_case:english_lowercase"}:
        import langdetect

        expected = args["language"] if identifier == "language:response_language" else "en"
        if not isinstance(expected, str) or not expected:
            raise InvalidTask("Instruction language must be nonempty text")
        components = []
        if identifier != "language:response_language":
            normalized = text.upper() if identifier.endswith("capital") else text.lower()
            components = [
                grade_ifeval_candidate(
                    IfevalSpec((Constraint(identifier, {}),), empty_output=EmptyOutputPolicy.GRADE), text
                ),
                grade_json_schema_candidate({"const": normalized}, text),
            ]
        try:
            detected = langdetect.detect(text)
        except langdetect.LangDetectException:
            detected = None
        components.append(grade_json_schema_candidate({"type": "string", "const": expected}, detected))
        return aggregate_rewards(components, expected_total=len(components), policy=Aggregation.ALL)
    schema = {"type": "string"}
    instance = text
    if identifier in {"keywords:existence", "keywords:forbidden_words"}:
        forbidden = identifier.endswith("forbidden_words")
        patterns = args["forbidden_words" if forbidden else "keywords"]
        if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
            raise InvalidTask("Instruction patterns must be strings")
        constraints = [
            {"pattern": "(?i)" + (r"\b" + pattern + r"\b" if forbidden else pattern)} for pattern in patterns
        ]
        schema["allOf"] = [{"not": item} for item in constraints] if forbidden else constraints
    elif identifier in {"length_constraints:number_words", "length_constraints:number_sentences"}:
        kind = "words" if identifier.endswith("words") else "sentences"
        schema = _count_schema(args["num_" + kind], args["relation"])
        namespace = original.check_following.__globals__
        tokenizer = (
            namespace["count_" + kind]
            if family == "IFEval"
            else getattr(namespace["instructions_util"], "count_" + kind)
        )
        instance = tokenizer(text)
    elif identifier == "keywords:frequency":
        schema = _count_schema(args["frequency"], args["relation"])
        try:
            parser = re.compile(args["keyword"], re.IGNORECASE)
        except re.error as error:
            raise InvalidTask("Invalid trusted keyword pattern") from error
        instance = len(parser.findall(text))
    elif identifier == "keywords:letter_frequency":
        schema = _count_schema(args["let_frequency"], args["let_relation"])
        instance = text.lower().count(args["letter"])
    elif identifier == "detectable_content:number_placeholders":
        schema = _count_schema(args["num_placeholders"], "at least")
        instance = len(re.findall(r"\[.*?\]", text))
    elif identifier == "detectable_format:number_highlighted_sections":
        schema = _count_schema(args["num_highlights"], "at least")
        single = [value.strip("*").strip() for value in re.findall(r"\*[^\n\*]*\*", text)]
        double = [
            value.removeprefix("**").removesuffix("**").strip() for value in re.findall(r"\*\*[^\n\*]*\*\*", text)
        ]
        instance = len([value for value in single + double if value])
    elif identifier == "change_case:capital_word_frequency":
        schema = _count_schema(args["capital_frequency"], args["capital_relation"])
        namespace = original.check_following.__globals__
        nltk = namespace["nltk"] if family == "IFEval" else namespace["instructions_util"].nltk
        instance = len([word for word in nltk.word_tokenize(text) if word.isupper()])
    elif identifier == "punctuation:no_comma":
        schema["not"] = {"pattern": ","}
    elif identifier == "length_constraints:number_paragraphs":
        parts = [part.strip() for part in re.split(r"\s?\*\*\*\s?", text)]
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        count = args["num_paragraphs"]
        _count_schema(count)
        schema = {"type": "array", "minItems": count, "maxItems": count, "items": {"type": "string", "minLength": 1}}
        instance = parts
    elif identifier == "length_constraints:nth_paragraph_first_word":
        parts = re.split(r"\n\n", text)
        nth = args["nth_paragraph"]
        if type(nth) is not int or nth < 1:
            raise InvalidTask("Paragraph index must be positive")
        words = parts[nth - 1].strip().split() if nth <= len(parts) else []
        first = re.split(r"""[.,?!'"]""", words[0].lstrip("'").lstrip('"'))[0].lower() if words else ""
        schema = {
            "type": "object",
            "properties": {
                "count": {**_count_schema(args["num_paragraphs"]), "minimum": nth},
                "first": {"const": args["first_word"]},
            },
        }
        instance = {"count": len([part for part in parts if part.strip()]), "first": first}
    elif identifier == "detectable_format:number_bullet_lists":
        instance = len(re.findall(r"^\s*\*[^\*].*$", text, re.MULTILINE)) + len(
            re.findall(r"^\s*-.*$", text, re.MULTILINE)
        )
        schema = _count_schema(args["num_bullets"])
    elif identifier == "detectable_format:multiple_sections":
        pattern = r"\s?" + args["section_spliter"] + r"\s?\d+\s?"
        schema = _count_schema(args["num_sections"], "at least")
        try:
            parser = re.compile(pattern)
        except re.error as error:
            raise InvalidTask("Invalid trusted section delimiter") from error
        instance = len(parser.split(text)) - 1
    elif identifier == "detectable_format:constrained_response":
        schema["anyOf"] = [{"pattern": re.escape(value)} for value in instruction._constrained_responses]
        instance = text.strip()
    elif identifier == "detectable_format:title":
        instance = [title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", text)]
        schema = {"type": "array", "contains": {"type": "string", "minLength": 1}}
    elif identifier == "detectable_format:json_format":
        value = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        schema = {"type": "object", "required": ["parsed"]}
        try:
            instance = {"parsed": json.loads(value, object_pairs_hook=unique_object)}
        except (ValueError, RecursionError):
            instance = {}
    elif identifier == "detectable_content:postscript":
        marker = args["postscript_marker"]
        pattern = {"P.P.S": r"\s*p\.\s?p\.\s?s.*$", "P.S.": r"\s*p\.\s?s\..*$"}.get(
            marker, r"\s*" + marker.lower() + r".*$"
        )
        schema["pattern"] = "(?m)" + pattern
        instance = text.lower()
    elif identifier == "startend:quotation":
        schema.update(minLength=2, pattern='(?s)^".*"$')
        instance = text.strip()
    elif identifier == "startend:end_checker":
        schema["pattern"] = re.escape(args["end_phrase"].strip().lower()) + r"\Z"
        instance = text.strip().strip('"').lower()
    elif identifier == "combination:repeat_prompt":
        prompt = args["prompt_to_repeat"]
        if not isinstance(prompt, str):
            raise InvalidTask("Prompt prefix must be text")
        schema["pattern"] = "^" + re.escape(prompt.strip().lower())
        instance = text.strip().lower()
    return grade_json_schema_candidate(schema, instance)
