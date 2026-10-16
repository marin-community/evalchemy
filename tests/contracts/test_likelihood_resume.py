"""Interrupted likelihood batches retain completed requests in FineStore."""

import pytest
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.contracts.finestore_resume import FineStoreResumeManager
from eval.contracts.sample_manifest import SampleManifest
from eval.resume import RunFingerprint
from eval.resume.lm_eval_native import resume_simple_evaluate


class _LikelihoodEndpoint(LM):
    def __init__(self, method, scores, fail_after=None, emit_partial=True):
        super().__init__()
        self.method = method
        self.scores = scores
        self.fail_after = fail_after
        self.emit_partial = emit_partial
        self.received = []

    def _respond(self, requests):
        self.received.extend(request.args for request in requests)
        # Completion order need not match the batch's input order.
        for index, request in enumerate(reversed(requests)):
            if index == self.fail_after:
                raise TimeoutError("endpoint disconnected")
            if self.emit_partial:
                self.cache_hook.add_partial(self.method, request.args, self.scores[request.args])
        return [self.scores[request.args] for request in requests]

    def loglikelihood(self, requests):
        return self._respond(requests)

    def loglikelihood_rolling(self, requests):
        return self._respond(requests)

    def generate_until(self, requests):
        return self._respond(requests)


def _fingerprint():
    return RunFingerprint.from_run_inputs(model_repo="model", task_name="task", task_data_digest="sha256:data")


def _resume_request_batch(root, model, evaluate):
    manifest = SampleManifest("task")
    result = resume_simple_evaluate(
        lambda model, **kwargs: evaluate(model),
        model=model,
        tasks=["task"],
        sample_manifest=manifest,
        resume_manager_factory=lambda task: FineStoreResumeManager(str(root), "run", task, _fingerprint()),
    )
    return result, manifest.generated_sample_count


@pytest.mark.parametrize(
    "method,arguments,values,evaluate",
    [
        pytest.param(
            "loglikelihood",
            [("question", "yes"), ("question", "no"), ("question", "yes"), ("other", "maybe")],
            [(-1.0, True), (-2.0, False), (-1.0, True), (-3.0, True)],
            lambda model, requests: model.loglikelihood(requests),
            id="answer-choices",
        ),
        pytest.param(
            "loglikelihood_rolling",
            [("first",), ("second",), ("first",), ("third",)],
            [-1.0, -2.0, -1.0, -3.0],
            lambda model, requests: model.loglikelihood_rolling(requests),
            id="rolling-documents",
        ),
    ],
)
def test_likelihood_resume_keeps_completed_requests_after_endpoint_failure(tmp_path, method, arguments, values, evaluate):
    scores = dict(zip(arguments, values))
    requests = [Instance(method, {}, args, index, metadata=("task", 0 if index < 2 else index, 1)) for index, args in enumerate(arguments)]
    interrupted = _LikelihoodEndpoint(method, scores, fail_after=2)

    def run(model):
        return evaluate(model, requests)

    with pytest.raises(TimeoutError):
        _resume_request_batch(tmp_path, interrupted, run)

    restored = FineStoreResumeManager(str(tmp_path), "run", "task", _fingerprint()).restore()
    assert len(restored) == 3
    retry = _LikelihoodEndpoint(method, scores, emit_partial=False)
    result, generated = _resume_request_batch(tmp_path, retry, run)

    assert result == values
    assert retry.received == [arguments[1]]
    assert generated == 3
    completed = _LikelihoodEndpoint(method, scores)
    assert _resume_request_batch(tmp_path, completed, run)[0] == result
    assert completed.received == []


def test_generation_resume_preserves_sampled_draws(tmp_path):
    request = Instance("generate_until", {}, ("question", {"do_sample": True}), 0, metadata=("task", 0, 2))
    requests = [request, request]

    class SampledEndpoint(_LikelihoodEndpoint):
        def generate_until(self, requests):
            self.received.extend(request.args for request in requests)
            return ["first draw", "second draw"]

    first = SampledEndpoint("generate_until", {})
    def run(model):
        return model.generate_until(requests)

    result, generated = _resume_request_batch(tmp_path, first, run)
    retry = SampledEndpoint("generate_until", {})

    assert result == ["first draw", "second draw"]
    assert generated == 1
    assert _resume_request_batch(tmp_path, retry, run)[0] == result
    assert retry.received == []
