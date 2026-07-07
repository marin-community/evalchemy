"""Small structured-result schemas for LLM judge calls.

These schemas intentionally use stdlib dataclasses instead of adding a direct
Pydantic dependency. They expose enough JSON schema for provider structured-output
calls and enough validation for offline tests and benchmark aggregation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar, Union


_VERDICTS = {"correct", "partial", "incorrect", "unjudgeable"}
T = TypeVar("T", bound="JudgeResult")


@dataclass
class JudgeRequest:
    task_name: str
    problem_id: str
    prompt: str
    response: str
    reference_answer: Optional[str] = None
    reference_solution: Optional[str] = None
    rubric: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None

    def to_messages(self, schema_name: str) -> List[Dict[str, str]]:
        """Render a provider-neutral grading prompt.

        DeepSeek JSON mode requires the word "json" in the prompt; keeping that
        instruction here is harmless for OpenAI structured-output calls and useful
        for any benchmark that does not supply a fully custom judge prompt.
        """

        system = self.system_prompt or (
            "You are a careful evaluation judge. Grade the submitted response against "
            "the problem, reference material, and rubric. Return only valid JSON that "
            f"matches the {schema_name} schema."
        )
        parts = [
            f"Task: {self.task_name}",
            f"Problem ID: {self.problem_id}",
            "Problem or grading prompt:",
            self.prompt,
            "Submitted response:",
            self.response,
        ]
        if self.reference_answer is not None:
            parts.extend(["Reference answer:", self.reference_answer])
        if self.reference_solution is not None:
            parts.extend(["Reference solution:", self.reference_solution])
        if self.rubric is not None:
            parts.extend(["Rubric JSON:", json.dumps(self.rubric, ensure_ascii=True, sort_keys=True)])
        parts.append("Return JSON only. Do not include markdown fences.")
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


@dataclass
class JudgeResult:
    score: float = 0.0
    verdict: str = "unjudgeable"
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    judge_model: Optional[str] = None
    judge_provider: Optional[str] = None
    judge_config_hash: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    provider_response_id: Optional[str] = None

    @classmethod
    def schema_name(cls) -> str:
        return cls.__name__

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "number", "description": "Overall score between 0 and 1."},
                "verdict": {"type": "string", "enum": sorted(_VERDICTS)},
                "reasoning": {"type": "string", "description": "Concise explanation of the grading decision."},
                "issues": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": ["number", "null"]},
            },
            "required": ["score", "verdict", "reasoning", "issues", "confidence"],
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if not isinstance(data, dict):
            raise ValueError("Judge result must be a JSON object")
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        result = cls(**kwargs)
        result.score = float(result.score)
        if result.verdict not in _VERDICTS:
            raise ValueError(f"Unsupported judge verdict: {result.verdict}")
        if result.confidence is not None:
            result.confidence = float(result.confidence)
        result.issues = list(result.issues or [])
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProofJudgeResult(JudgeResult):
    logical_validity: float = 0.0
    completeness: float = 0.0
    rigor: float = 0.0
    final_claim_established: bool = False
    fatal_gap: Optional[str] = None
    hallucinated_theorem: bool = False
    rubric_points: List[Dict[str, Union[str, float, None]]] = field(default_factory=list)

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        schema = super().json_schema()
        schema["properties"].update(
            {
                "logical_validity": {"type": "number"},
                "completeness": {"type": "number"},
                "rigor": {"type": "number"},
                "final_claim_established": {"type": "boolean"},
                "fatal_gap": {"type": ["string", "null"]},
                "hallucinated_theorem": {"type": "boolean"},
                "rubric_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "score": {"type": ["number", "null"]},
                            "explanation": {"type": "string"},
                        },
                        "required": ["name", "score", "explanation"],
                    },
                },
            }
        )
        schema["required"] = schema["required"] + [
            "logical_validity",
            "completeness",
            "rigor",
            "final_claim_established",
            "fatal_gap",
            "hallucinated_theorem",
            "rubric_points",
        ]
        return schema

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofJudgeResult":
        result = super().from_dict(data)
        result.logical_validity = float(result.logical_validity)
        result.completeness = float(result.completeness)
        result.rigor = float(result.rigor)
        result.final_claim_established = bool(result.final_claim_established)
        result.hallucinated_theorem = bool(result.hallucinated_theorem)
        result.rubric_points = _normalize_rubric_points(result.rubric_points)
        return result


def _normalize_rubric_points(value: Any) -> List[Dict[str, Union[str, float, None]]]:
    if value is None:
        return []
    if isinstance(value, dict):
        normalized = []
        for name, raw in sorted(value.items()):
            score = float(raw) if isinstance(raw, (int, float)) else None
            normalized.append({"name": str(name), "score": score, "explanation": str(raw)})
        return normalized
    if isinstance(value, list):
        normalized = []
        for item in value:
            if not isinstance(item, dict):
                normalized.append({"name": str(item), "score": None, "explanation": str(item)})
                continue
            normalized.append(
                {
                    "name": str(item.get("name", "")),
                    "score": float(item["score"]) if isinstance(item.get("score"), (int, float)) else None,
                    "explanation": str(item.get("explanation", "")),
                }
            )
        return normalized
    return [{"name": "rubric_points", "score": None, "explanation": str(value)}]
