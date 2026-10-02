"""Exercise judge transport validation through the shared Judge candidate API."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.graders.answer_equivalence import JudgeConfig, JudgeLabel
from eval.graders.simpleqa import SimpleQARequest
from eval.graders.verifyit_judges import judge_simpleqa
from verifyit.grade import InvalidTask


@pytest.fixture
def judge_server():
    state = {"label": "A", "fault": None, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            bad = any(marker in body["messages"][0]["content"] for marker in ("Question: bad", "Question:\nbad"))
            fault = state["fault"] if bad else None
            message = {"role": "assistant", "content": state["label"]}
            choice = {"index": 0, "finish_reason": "stop", "message": message}
            if fault == "truncated":
                choice["finish_reason"] = "length"
            elif fault == "refusal":
                message["refusal"] = "refused"
            elif fault == "retry" and body.get("max_completion_tokens", body.get("max_tokens")) == 128:
                message["content"] = ""
                choice["finish_reason"] = "length"
            elif fault == "invalid":
                message["content"] = "A because correct"
            raw = json.dumps(
                {"id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture", "choices": [choice]}
            )
            if fault == "duplicate":
                raw = raw.replace('"content": "A"', '"content": "B", "content": "A"')
            data = raw.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, JudgeConfig("fixture", f"http://127.0.0.1:{server.server_port}/v1", "fixture")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("label", "expected"), [("A", JudgeLabel.CORRECT), ("B", JudgeLabel.INCORRECT), ("C", JudgeLabel.NOT_ATTEMPTED)]
)
def test_judge_scores_completed_labels(judge_server, label, expected):
    state, config = judge_server
    state["label"] = label
    results = asyncio.run(judge_simpleqa([SimpleQARequest("good", "Paris", "candidate")], config))
    assert results[0].label == expected
    assert results[0].raw == label


def test_judge_retries_incomplete_completion_with_shared_token_budget(judge_server):
    state, config = judge_server
    state["fault"] = "retry"
    results = asyncio.run(judge_simpleqa([SimpleQARequest("bad", "Paris", "candidate")], config))
    assert results[0].label == JudgeLabel.CORRECT
    assert [request["max_completion_tokens"] for request in state["requests"]] == [128, 2048]


@pytest.mark.parametrize("fault", ["truncated", "refusal", "duplicate", "invalid"])
def test_invalid_judge_completion_aborts_batch_after_valid_result(judge_server, fault):
    state, config = judge_server
    state["fault"] = fault
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(
            judge_simpleqa(
                [SimpleQARequest("good", "Paris", "candidate"), SimpleQARequest("bad", "Paris", "candidate")],
                config,
                num_workers=1,
            )
        )
    assert len(state["requests"]) == (3 if fault == "truncated" else 2)


def test_invalid_reference_rejects_entire_batch_before_requests(judge_server):
    state, config = judge_server
    with pytest.raises(InvalidTask):
        asyncio.run(
            judge_simpleqa(
                [SimpleQARequest("good", "Paris", "candidate"), SimpleQARequest("bad", " ", "candidate")], config
            )
        )
    assert state["requests"] == []


@pytest.mark.parametrize(("label", "accuracy"), [("correct", 1.0), ("incorrect", 0.0), ("not_attempted", 0.0)])
def test_financebench_source_labels_drive_metrics(judge_server, label, accuracy):
    state, config = judge_server
    state["label"] = label
    benchmark = FinanceBenchBenchmark(
        annotator_model=config.model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        verifyit_enabled=True,
    )
    result = benchmark.evaluate_responses({"examples": [{"question": "good", "answer": "$5", "model_output": "$5"}]})
    assert result["accuracy"] == accuracy
    assert result["num_judged"] == 1
    assert result["examples"][0]["judge_label"] == label


def test_financebench_judge_failure_aborts_instead_of_excluding_failed_sample(judge_server):
    state, config = judge_server
    state.update(label="correct", fault="truncated")
    benchmark = FinanceBenchBenchmark(
        annotator_model=config.model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        verifyit_enabled=True,
    )
    examples = [{"question": question, "answer": "$5", "model_output": "$5"} for question in ("good", "bad")]
    with pytest.raises(RuntimeError):
        benchmark.evaluate_responses({"examples": examples})
    assert all("judge_label" not in example for example in examples)


def test_financebench_missing_reference_is_not_stringified(judge_server):
    state, config = judge_server
    state["label"] = "correct"
    benchmark = FinanceBenchBenchmark(
        annotator_model=config.model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        verifyit_enabled=True,
    )
    with pytest.raises(InvalidTask):
        benchmark.evaluate_responses({"examples": [{"question": "good", "answer": None, "model_output": "None"}]})
    assert not state["requests"]


@pytest.mark.parametrize("candidate", ["  ", None, 42])
def test_empty_candidate_cannot_receive_positive_judge_credit(judge_server, candidate):
    state, config = judge_server
    state["label"] = "A"
    results = asyncio.run(judge_simpleqa([SimpleQARequest("good", "Paris", candidate)], config))
    assert results[0].label == JudgeLabel.NOT_ATTEMPTED
    assert results[0].raw == ""
    assert state["requests"] == []


def test_olympiad_judge_accepts_textual_equation_outside_math_parser(judge_server):
    state, config = judge_server
    state["label"] = "correct"
    benchmark = OlympiadBenchBenchmark(
        annotator_model=config.model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        n_repeat=1,
        verifyit_enabled=True,
    )
    equation = r"\omega(t)=\frac{e c^{2} B}{E_{0}}(1+\frac{e^{4} B^{2}}{6 \pi \epsilon_{0} m^{4} c^{5}} E_{0} t)"
    result = benchmark.evaluate_responses(
        {"examples": [{"problem": "Find the angular frequency.", "answer": equation, "model_answer": "a textual solution"}]}
    )
    assert result["accuracy"] == 1
    assert result["num_graded_by_minerva"] == 0
    assert result["num_judged_by_llm"] == 1
    assert len(state["requests"]) == 1
    assert equation.replace("\\", "\\\\") in state["requests"][0]["messages"][0]["content"]


@pytest.mark.parametrize(("question", "answer"), [(None, "42"), ("good", [None])])
def test_olympiad_invalid_raw_reference_aborts_before_judge(judge_server, question, answer):
    state, config = judge_server
    state["label"] = "correct"
    benchmark = OlympiadBenchBenchmark(
        annotator_model=config.model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        n_repeat=1,
        verifyit_enabled=True,
    )
    with pytest.raises(InvalidTask):
        benchmark.evaluate_responses(
            {"examples": [
                {"problem": "good", "answer": "42", "model_answer": "42"},
                {"problem": question, "answer": answer, "model_answer": None},
            ]}
        )
    assert state["requests"] == []
