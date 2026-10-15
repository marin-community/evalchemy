from eval.sources.common import conversation_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from dataclasses import dataclass
from typing import Optional
import os
import logging
from pathlib import Path


from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.task import BaseBenchmark
from fastchat.llm_judge.common import load_questions, load_model_answers, load_judge_prompts, NEED_REF_CATS, temperature_config


@dataclass
class MTBenchConfig:
    """Configuration for MTBench evaluation."""

    question_begin: Optional[int] = None
    question_end: Optional[int] = None
    max_new_token: int = 1024
    judge_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fastchat", "llm_judge", "data", "judge_prompts.jsonl"
    )
    judge_model: str = "gpt-4o-mini-2024-07-18"
    mode: str = "single"


class MTBenchBenchmark(BaseBenchmark):
    """
    MTBench benchmark for evaluating multi-turn chat capabilities.
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # Can also be anthropic

    def __init__(
        self,
        base_path: str = os.path.dirname(os.path.abspath(__file__)),
        config: Optional[MTBenchConfig] = None,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MTBench benchmark.

        Args:
            base_path: Base directory for bundled MTBench data
            config: MTBench configuration object
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        base_path = Path(base_path)
        if annotator_model == AUTO_ANNOTATOR_MODEL:
            annotator_model = "gpt-4"
        if config:
            print(f"Warning: Overwriting config.judge_model = {annotator_model} ")
            config.judge_model = annotator_model
        self.config = config or MTBenchConfig(judge_model=annotator_model)
        self.config.max_new_token = max_tokens
        self.debug = debug

        # Setup paths
        self.data_path = base_path / "fastchat/llm_judge/data/mt_bench"
        self.question_file = self.data_path / "question.jsonl"
        self.ref_answer_dir = self.data_path / "reference_answer"
        self.judge_model = self.config.judge_model

    REQUIRES_JUDGE = True
    METRICS = ("Turn 1", "Turn 2", "Average")
    PRIMARY_METRIC = "Average"
    REDUCER = ReducerSpec(trial_metric=None)
    SOURCE = SourceRef("lmsys/mt-bench", "test")

    def iter_specs(self, options):
        if self.config.mode != "single":
            raise ValueError("MTBench pairwise mode requires an explicit comparison source")
        questions = load_questions(self.question_file, self.config.question_begin, self.config.question_end)
        if self.debug:
            questions = questions[:2]
        templates = load_judge_prompts(self.config.judge_file)
        references = load_model_answers(self.ref_answer_dir)
        def reference(row):
            reference_based = row["category"] in NEED_REF_CATS
            key = "single-math-v1" if reference_based else "single-v1"
            answers = references[self.config.judge_model][row["question_id"]]["choices"][0]["turns"] if reference_based else []
            return {"templates": [templates[key], templates[key + "-multi-turn"]], "answers": answers}
        return conversation_specs(
            questions, options, turns=lambda row: row["turns"], target=lambda row: row["turns"],
            grader="mtbench", reference=reference,
            generation=lambda row: {"max_gen_toks": self.config.max_new_token,
                                    "do_sample": temperature_config.get(row["category"], 0.7) >= 1e-4,
                                    "temperature": temperature_config.get(row["category"], 0.7)},
        )
