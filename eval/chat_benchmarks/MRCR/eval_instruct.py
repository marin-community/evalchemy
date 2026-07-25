import json
import logging
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

# MRCR (Multi-round Co-reference Resolution) -- openai/mrcr on HuggingFace.
# A long-context multi-needle benchmark: the model is given a very long multi-turn
# conversation (16K-5.6M chars) with N (2/4/8) identical requests hidden among
# distractors, and must return the i-th instance of a specific request, prepended
# with a random 10-char hash. Introduced by Michelangelo
# (arxiv:2409.12640); open-sourced by OpenAI at
# https://huggingface.co/datasets/openai/mrcr. 2,400 rows total.
DATASET_NAME = "openai/mrcr"

# Default chars-per-token ratio for the max_context_tokens filter. The dataset
# ships a precomputed `n_chars` field (not a token count), and the served model's
# tokenizer is not available on the endpoint-eval path, so we estimate tokens as
# n_chars / chars_per_token. 4.0 is a standard English-text heuristic; override
# via `chars_per_token` if you know your model's ratio.
DEFAULT_CHARS_PER_TOKEN = 4.0


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Official MRCR grader (from the openai/mrcr dataset card).

    Returns 0.0 if the response does not start with the required prepend hash;
    otherwise strips the prepend from both and returns the
    `difflib.SequenceMatcher` ratio between the stripped response and the
    stripped gold answer.
    """
    response = str(response)
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = str(answer).removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


class MRCRBenchmark(BaseBenchmark):
    """
    MRCR (Multi-round Co-reference Resolution) benchmark from openai/mrcr.

    A long-context multi-needle eval (arxiv:2409.12640, "Michelangelo: Long
    Context Evaluations Beyond Haystacks via Latent Structure Queries"): given a
    very long multi-turn conversation (16K-5.6M chars) with N identical requests
    (needles) hidden among distractors, the model must return the i-th instance
    of a specific request, prepended with a random 10-char hash.

    Grading is the official SequenceMatcher ratio: 0.0 if the prepend hash is
    missing, else the ratio between the prepend-stripped response and the
    prepend-stripped gold answer. Reported as the mean ratio overall and broken
    down by n_needles (2/4/8).

    Link: https://huggingface.co/datasets/openai/mrcr
    """

    def __init__(
        self,
        n_needles: Optional[Union[int, List[int]]] = None,
        max_context_tokens: Optional[int] = None,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        cache_dir: Optional[str] = os.environ.get("HF_HUB_CACHE"),
    ):
        """
        Initialize MRCR benchmark.

        Args:
            n_needles: Filter to a specific needle count (2, 4, or 8) or a list of
                them. None (default) = use all rows (2,400 across 2/4/8-needle).
            max_context_tokens: Drop rows whose estimated token count exceeds this
                (estimated = n_chars / chars_per_token). None (default) = no filter,
                so rows up to ~5.6M chars (~1.4M tokens) are kept -- set this to
                your model's context window to avoid sending overlong prompts.
            chars_per_token: Chars-per-token ratio for the max_context_tokens
                filter (default 4.0, an English-text heuristic). Ignored when
                max_context_tokens is None.
            debug: If set, only evaluate on 2 examples.
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234]
                for lm-eval-harness.
            max_tokens: max_new_tokens for generation. The MRCR response is a
                single piece of text (the requested needle), so 4096 by default.
            logger: Optional logger instance.
            system_instruction: Optional system instruction prepended to the
                prompt. The dataset's prompt is already a complete multi-turn
                conversation, so leave this None unless you know what you are doing.
            cache_dir: HF datasets cache dir (defaults to $HF_HUB_CACHE).
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        if isinstance(n_needles, int):
            n_needles = [n_needles]
        self.n_needles = n_needles
        self.max_context_tokens = max_context_tokens
        self.chars_per_token = chars_per_token
        self.debug = debug
        self.seed = seed
        # Single piece of retrieved text; deterministic retrieval.
        self.max_new_tokens = max_tokens
        self.cache_dir = cache_dir

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load MRCR rows from openai/mrcr, filtered by n_needles and token budget."""
        self.logger.info(f"Loading {DATASET_NAME}...")
        ds = load_dataset(DATASET_NAME, cache_dir=self.cache_dir)["train"]
        questions = list(ds)

        if self.n_needles is not None:
            before = len(questions)
            questions = [q for q in questions if int(q["n_needles"]) in self.n_needles]
            self.logger.info(
                f"n_needles filter {self.n_needles}: {before} -> {len(questions)} rows"
            )

        if self.max_context_tokens is not None:
            char_budget = int(self.max_context_tokens * self.chars_per_token)
            before = len(questions)
            questions = [q for q in questions if int(q["n_chars"]) <= char_budget]
            self.logger.info(
                f"max_context_tokens filter ({self.max_context_tokens} ~= {char_budget} chars): "
                f"{before} -> {len(questions)} rows"
            )

        if self.debug:
            questions = questions[:2]
            self.logger.info(
                f"Debug mode enabled. Using only {len(questions)} questions."
            )

        self.logger.info(f"Loaded {len(questions)} MRCR questions from {DATASET_NAME}")
        return questions

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses by replaying each row's full multi-turn prompt.

        Args:
            model: Language model.

        Returns:
            Dictionary containing generated responses and examples,
            or None for non-primary ranks.
        """
        examples = self.load_questions()

        all_instances: List[Instance] = []
        for idx, example in enumerate(examples):
            messages = self._parse_prompt(example["prompt"])
            templated_messages = self._prepare_messages(messages, model)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.0,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
            )

        self.logger.info("Generating responses for MRCR...")
        outputs = self.compute(model, all_instances)

        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)
            example["score"] = grade(
                output, example["answer"], example["random_string_to_prepend"]
            )

        return {"examples": examples}

    @staticmethod
    def _parse_prompt(prompt: str) -> List[Dict[str, str]]:
        """Parse a row's `prompt` field (a JSON-encoded list of chat messages)."""
        messages = json.loads(prompt)
        if not isinstance(messages, list):
            raise ValueError(
                f"MRCR prompt is not a list of messages (got {type(messages).__name__})"
            )
        return messages

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate per-row SequenceMatcher ratios, overall and by n_needles."""
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)

        if total == 0:
            results.update(
                {"num_total": 0, "mean_score": 0.0, "scores_by_n_needles": {}}
            )
            return results

        scores = [float(ex["score"]) for ex in examples]
        mean_score = sum(scores) / total

        by_n: Dict[int, List[float]] = {}
        for ex, s in zip(examples, scores):
            by_n.setdefault(int(ex["n_needles"]), []).append(s)
        scores_by_n_needles = {
            n: {"count": len(ss), "mean_score": sum(ss) / len(ss)}
            for n, ss in sorted(by_n.items())
        }

        results.update(
            {
                "num_total": total,
                "mean_score": mean_score,
                "scores_by_n_needles": scores_by_n_needles,
            }
        )
        return results

    def extract_answer(self, output: str) -> str:
        """MRCR has no structural answer marker -- the full response IS the answer.

        The official grader handles the prepend-hash check and stripping; we keep
        the raw response as model_answer for the sample record.

        Args:
            output: Model-generated response text.

        Returns:
            The response text unchanged.
        """
        return str(output)

    def _sample_prompt(self, example: Dict[str, Any]) -> str:
        """Raw dataset `prompt` field (JSON-encoded messages) for the sample record."""
        return str(example.get("prompt", ""))
