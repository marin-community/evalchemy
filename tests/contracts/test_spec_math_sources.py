"""Offline source fixtures for math and instruction benchmark item protocols."""

import json

from eval.chat_benchmarks.AIW.eval_instruct import AIWBenchmark
from eval.chat_benchmarks.HLE.eval_instruct import HLESubsetBenchmark
from eval.chat_benchmarks.HMMT.eval_instruct import HMMTBenchmark
from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.chat_benchmarks.JEEBench.eval_instruct import JEEBenchBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions


def _options(task):
    return SourceOptions(task, SourceRef("fixture"), seed=7)


def test_aiw_selects_target_ids_before_expanding_trials(tmp_path):
    data = tmp_path / "aiw.json"
    data.write_text(json.dumps([
        {"id": 1, "prompt": "skip", "right_answer": 0},
        {"id": 577, "prompt": "keep", "right_answer": 42},
    ]))

    specs = list(AIWBenchmark(data_file=data, n_trials=2).iter_specs(_options("AIW")))

    assert [spec.source_id for spec in specs] == [577, 577]
    assert [spec.trial for spec in specs] == [0, 1]
    assert specs[0].target == "42"
    assert specs[0].requests[0].generation["temperature"] == 1.0


def test_aiw_debug_selects_target_rows_before_limiting(tmp_path):
    data = tmp_path / "aiw.json"
    data.write_text(json.dumps([
        {"id": 1, "prompt": "skip", "right_answer": 0},
        {"id": 577, "prompt": "first", "right_answer": 42},
        {"id": 580, "prompt": "second", "right_answer": 43},
    ]))

    specs = list(AIWBenchmark(data_file=data, debug=True, n_trials=1).iter_specs(_options("AIW")))

    assert [spec.source_id for spec in specs] == [577, 580]


def test_hle_and_hmmt_preserve_prompt_and_repeat_protocols(monkeypatch):
    hle = HLESubsetBenchmark()
    monkeypatch.setattr(hle, "load_questions", lambda: [{"id": "h", "answer": "B", "answer_type": "multipleChoice", "image": "", "question": "Choose B"}])
    hle_specs = list(hle.iter_specs(_options("HLE")))
    assert [spec.trial for spec in hle_specs] == [0, 1, 2]
    assert hle_specs[0].requests[0].prompt == [{"role": "user", "content": "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text.\nChoose B"}]

    hmmt = HMMTBenchmark()
    monkeypatch.setattr(hmmt, "load_questions", lambda: [{"id": "m", "problem": "Find x", "answer": "3"}])
    hmmt_specs = list(hmmt.iter_specs(_options("HMMT")))
    assert len(hmmt_specs) == 10
    assert hmmt_specs[0].requests[0].prompt[0]["content"].endswith("\\boxed{}.\nAnswer:")


def test_jeebench_and_ifeval_specs_keep_source_selection(tmp_path, monkeypatch):
    jee = JEEBenchBenchmark()
    monkeypatch.setattr(jee, "load_questions", lambda: [{"index": 3, "type": "Integer", "question": "\n\nFind 2.", "gold": "2", "subject": "math"}])
    jee_specs = list(jee.iter_specs(_options("JEEBench")))
    assert len(jee_specs) == 3
    assert jee_specs[0].grader.name == "jee"
    assert "Problem: Find 2." in jee_specs[0].requests[0].prompt[0]["content"]

    data = tmp_path / "input_data.jsonl"
    rows = [{"key": index, "prompt": f"prompt {index}", "instruction_id_list": ["length_constraints:number_words"], "kwargs": [{}]} for index in range(12)]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ifeval_specs = list(IFEvalBenchmark(data_dir=str(tmp_path), start_idx=10, end_idx=12).iter_specs(_options("IFEval")))
    assert [spec.source_id for spec in ifeval_specs] == [10, 11]
    assert ifeval_specs[0].grader.reference == {"protocol": "ifeval"}
