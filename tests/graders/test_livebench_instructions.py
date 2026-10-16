"""LiveBench instruction questions retain their native partial-credit policy."""

from copy import deepcopy

import pytest

from eval.chat_benchmarks.LiveBench.livebench.process_results.instruction_following.utils import instruction_following_process_results
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.conversations import livebench


@pytest.mark.parametrize("task", ["simplify", "story_generation", "paraphrase", "summarize"])
@pytest.mark.parametrize(("response", "expected"), [("amber", 100), ("amber, violet", 25), ("violet, blue", 0), ("", 0)])
def test_livebench_instruction_score_matches_native_policy(tmp_path, monkeypatch, task, response, expected):
    question = {
        "question_id": 17, "task": task, "category": "instruction_following",
        "turns": ["Include amber and do not use a comma."],
        "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
        "kwargs": [{"keywords": ["amber"], "unused": None}, {}],
    }
    answer = {"question_id": 17, "choices": [{"turns": [response]}]}
    monkeypatch.chdir(tmp_path)
    # The corpus grader writes its strict-check details under this native path.
    (tmp_path / "eval/chat_benchmarks/LiveBench/data/live_bench/instruction_following" / task / "model_judgment").mkdir(parents=True)
    [native] = instruction_following_process_results([deepcopy(question)], {"candidate": {17: answer}}, task, "candidate")
    assert 100 * native["score"] == expected

    unchanged = deepcopy(question)
    spec = EvalSpec("LiveBench", SourceRef("fixture"), 17, 0, (), GraderSpec("livebench"), doc=question)
    grade = livebench(spec, [response])

    assert grade.metrics == {"accuracy": expected}
    assert spec.doc == unchanged
