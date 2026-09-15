"""Canonical serialization for Evalchemy result artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lm_eval.utils import handle_non_serializable


def safe_artifact_name(value: str) -> str:
    """Return a filesystem- and object-key-safe Evalchemy artifact name."""
    return re.sub(r"[^\w.-]", "_", value) or "task"


def results_json(results: Mapping[str, Any]) -> str:
    """Serialize Evalchemy's aggregate results artifact."""
    return json.dumps(
        results,
        indent=2,
        default=handle_non_serializable,
        ensure_ascii=False,
    )


def samples_jsonl(samples: Sequence[Mapping[str, Any]]) -> str:
    """Serialize Evalchemy's per-task sample artifact."""
    return "".join(
        json.dumps(sample, default=handle_non_serializable, ensure_ascii=False) + "\n" for sample in samples
    )
