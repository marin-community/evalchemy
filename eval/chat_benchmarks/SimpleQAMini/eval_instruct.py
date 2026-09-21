"""Deterministic 500-question SimpleQA subset."""

import logging
import random
from typing import Any, Dict, List, Optional

from eval.chat_benchmarks.SimpleQA.eval_instruct import DEFAULT_DATA_FILE, SimpleQABenchmark

MINI_DATASET_SIZE = 500
MINI_SAMPLE_SEED = 0


class SimpleQAMiniBenchmark(SimpleQABenchmark):
    """Evaluate OpenAI's seeded 500-question SimpleQA mini subset."""

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        debug: bool = False,
        seed: tuple[int, int, int, int] = (0, 1234, 1234, 1234),
        max_tokens: int = 512,
        annotator_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(
            data_file=data_file,
            debug=debug,
            seed=seed,
            max_tokens=max_tokens,
            annotator_model=annotator_model,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
            logger=logger,
            system_instruction=system_instruction,
        )

    def benchmark_size(self) -> int:
        return MINI_DATASET_SIZE

    def load_questions(self) -> List[Dict[str, Any]]:
        """Select the same seeded subset used by OpenAI's reference evaluator."""
        questions = random.Random(MINI_SAMPLE_SEED).sample(self._load_source_questions(), MINI_DATASET_SIZE)
        if self.debug:
            return questions[:2]
        return questions
