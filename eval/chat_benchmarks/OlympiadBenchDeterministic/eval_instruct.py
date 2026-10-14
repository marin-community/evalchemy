"""OlympiadBench subset scored only by Minerva/SymPy equivalence."""

import logging
from typing import List, Optional

from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark


class OlympiadBenchDeterministicBenchmark(OlympiadBenchBenchmark):
    """Use the historical 30-row subset without the LLM-judge fallback.

    Scores are not directly comparable with judge-backed OlympiadBench runs.
    """

    REQUIRES_JUDGE = False

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[List[int] | str] = None,
        n_repeat: int = 1,
        verifyit_enabled: bool = False,
        verifyit_policy: str = "source_first_box_dollar_alternatives_meaningful_math_failclosed_v1",
        verifyit_timeout: float = 30,
    ):
        self.verifyit_policy = verifyit_policy
        super().__init__(
            debug=debug,
            verifyit_enabled=verifyit_enabled,
            verifyit_timeout=verifyit_timeout,
            seed=seed,
            max_tokens=max_tokens,
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
            n_repeat=n_repeat,
        )

    def evaluate_responses(self, results):
        if not self.verifyit_enabled or results is None:
            return super().evaluate_responses(results)
        from eval.graders.verifyit_olympiad_deterministic import remaining_budget, stage_results, total_deadline

        with total_deadline(self.verifyit_timeout):
            staged = stage_results(results, num_samples=self.num_samples, repeats=self.n_repeat)
            scored = super().evaluate_responses(staged)
            remaining_budget(self.verifyit_timeout)
            results.update(scored)
        return results

    def _grade_model_answers(self, examples, answers_by_example):
        if not self.verifyit_enabled:
            return super()._grade_model_answers(examples, answers_by_example)
        from eval.graders.verifyit_olympiad_deterministic import grade_batch

        count = self.num_samples if self.num_samples > 1 else self.n_repeat
        batches = grade_batch(examples, count, policy=self.verifyit_policy, timeout=self.verifyit_timeout)
        correct = []
        for example, records in zip(examples, batches, strict=True):
            correct.append([bool(record["reward"]) for record in records])
            if count > 1:
                example["model_answers"] = [record["candidate"] for record in records]
            if count == 1:
                example["model_answer"] = records[0]["candidate"]
            example["verifyit_grades"] = [{key: value for key, value in record.items() if key != "candidate"}
                                         for record in records]
            grades = [{"method": "minerva", "judge_label": None, "judge_raw": None} for _ in records]
            example["equivalence_grades"] = grades
            if count == 1:
                example.update(grades[0])
        return correct, {"num_graded_by_minerva": sum(map(len, batches)), "num_judged_by_llm": 0,
                         "num_judge_failed": 0, "judge_model": None}
