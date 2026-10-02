"""Prepare source instruction observations for existing core comparisons."""

import csv
import io
import string
from collections import Counter

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


IFBENCH_IDS = {
    "repeat:repeat_change",
    "repeat:repeat_simple",
    "count:numbers",
    "words:keywords_specific_position",
    "count:keywords_multiple",
    "words:prime_lengths",
    "format:newline",
    "sentence:keyword",
    "count:word_count_range",
    "custom:multiples",
    "words:repeats",
    "format:options",
    "repeat:repeat_span",
    "words:words_position",
    "format:output_template",
    "custom:character_reverse",
    "custom:csv_city",
    "format:no_whitespace",
    "count:unique_word_count",
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
    if family == "IFBench":
        return _grade_ifbench_instruction(identifier, instruction, original, text)
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


def _grade_ifbench_instruction(identifier, instruction, original, text):
    if identifier not in IFBENCH_IDS:
        return None
    args = instruction.get_instruction_args() or {}
    args = {key: int(value) if type(value) is float and value.is_integer() else value for key, value in args.items()}
    namespace = original.check_following.__globals__
    punctuation = string.punctuation
    schema = {"type": "string"}
    instance = text
    if identifier == "count:word_count_range":
        low, high = args["min_words"], args["max_words"]
        _count_schema(low)
        _count_schema(high)
        if low > high:
            raise InvalidTask("Reversed trusted word-count range")
        schema = {"type": "integer", "minimum": low, "maximum": high}
        instance = namespace["instructions_util"].count_words(text)
    elif identifier == "count:unique_word_count":
        schema = _count_schema(args["N"], "at least")
        instance = len({word.strip(punctuation + " ") for word in text.lower().split()})
    elif identifier == "count:numbers":
        schema = _count_schema(args["N"])
        instance = len(re.findall(r"\d+", text.translate(str.maketrans("", "", punctuation))))
    elif identifier == "words:repeats":
        _count_schema(args["small_n"])
        schema = {"type": "array", "items": {"type": "integer", "maximum": args["small_n"]}}
        words = text.lower().translate(str.maketrans("", "", punctuation)).split()
        instance = list(Counter(words).values())
    elif identifier == "format:options":
        options = instruction._options
        if not instruction._strict:
            options = [option.strip(punctuation + " ").lower() for option in options]
            instance = text.strip(punctuation + " ").lower()
        if not options or any(not isinstance(option, str) or not option for option in options):
            raise InvalidTask("Trusted instruction options must contain nonempty labels")
        schema["enum"] = options
    elif identifier == "format:newline":
        value = text.translate(str.maketrans("", "", punctuation))
        schema = _count_schema(len(value.strip().split()))
        instance = len([line for line in value.strip().split("\n") if line != ""])
    elif identifier in {"sentence:keyword", "words:keywords_specific_position"}:
        position = args["N"] if identifier == "sentence:keyword" else args["n"]
        _count_schema(position)
        if position < 1:
            raise InvalidTask("Sentence index must be positive")
        sentences = namespace["instructions_util"].split_into_sentences(text)
        sentence = sentences[position - 1] if position <= len(sentences) else None
        if identifier == "sentence:keyword":
            schema["pattern"] = r"(?i)\b" + re.escape(args["word"]) + r"\b"
            instance = sentence
        else:
            word_position = args["m"]
            _count_schema(word_position)
            if word_position < 1:
                raise InvalidTask("Word index must be positive")
            words = namespace["_word_tokens_without_punctuation"](sentence) if sentence is not None else []
            schema["const"] = args["keyword"].lower()
            instance = words[word_position - 1].lower() if word_position <= len(words) else None
    elif identifier == "count:keywords_multiple":
        schema = {"type": "array", "const": [1, 2, 3, 5, 7]}
        instance = [text.lower().count(args[f"keyword{index}"].lower()) for index in range(1, 6)]
    elif identifier == "words:words_position":
        words = namespace["instructions_util"].nltk.word_tokenize(text)
        last = -3 if words and words[-1] in punctuation else -2
        instance = [words[1].lower(), words[last].lower()] if len(words) >= max(2, -last) else []
        schema = {"type": "array", "minItems": 2, "items": {"const": args["keyword"].lower()}}
    elif identifier == "repeat:repeat_change":
        schema = {
            "type": "object",
            "properties": {
                "whole": {"not": {"const": args["prompt_to_repeat"]}},
                "tail": {"const": " ".join(args["prompt_to_repeat"].split()[1:])},
            },
        }
        instance = {"whole": text, "tail": " ".join(text.split()[1:])}
    elif identifier in {"repeat:repeat_simple", "repeat:repeat_span"}:
        reference = instruction._description_pattern
        if identifier == "repeat:repeat_span":
            start, end = args["n_start"], args["n_end"]
            _count_schema(start)
            _count_schema(end)
            if end < start:
                raise InvalidTask("Reversed trusted repeat span")
            reference = args["prompt_to_repeat"][start : end + 1]
        schema["const"] = reference.strip().lower()
        instance = text.strip().lower()
    elif identifier == "format:output_template":
        schema["allOf"] = [
            {"pattern": re.escape(marker)} for marker in ("My Answer:", "My Conclusion:", "Future Outlook:")
        ]
    elif identifier == "format:no_whitespace":
        schema["not"] = {"pattern": r"\s"}
    elif identifier == "custom:multiples":
        schema = {"type": "array", "const": [str(value) for value in range(14, 51, 7)]}
        instance = re.findall(r"\d+", text.replace(",", ", "))
    elif identifier == "custom:character_reverse":
        schema["pattern"] = "elgae dlab"
        instance = text.lower()
    elif identifier == "custom:csv_city":
        schema = {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "prefixItems": [{"const": ["ID", "Country", "City", "Year", "Count"]}],
            "items": {"type": "array", "minItems": 5, "maxItems": 5},
        }
        try:
            instance = list(csv.reader(io.StringIO(text)))
        except csv.Error:
            instance = None
    elif identifier == "words:prime_lengths":
        schema = {
            "type": "array",
            "items": {
                "enum": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            },
        }
        instance = [len(word) for word in text.translate(str.maketrans("", "", punctuation)).split()]
    return grade_json_schema_candidate(schema, instance)
