"""SWE-bench closes ordinary Docker clients after grading or harness failure."""

import sys
from types import ModuleType

import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.swebench import swe_patch


class DockerClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def swebench_boundary(monkeypatch, tmp_path):
    client = DockerClient()
    docker = ModuleType("docker")
    docker.from_env = lambda: client
    constants = ModuleType("swebench.harness.constants")
    constants.KEY_INSTANCE_ID = "instance_id"
    constants.KEY_MODEL = "model_name_or_path"
    constants.KEY_PREDICTION = "model_patch"
    constants.APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
    constants.LOG_INSTANCE = "run_instance.log"
    constants.RUN_EVALUATION_LOG_DIR = tmp_path
    test_spec = ModuleType("swebench.harness.test_spec.test_spec")
    test_spec.make_test_spec = lambda instance, **_kwargs: instance
    harness = ModuleType("swebench.harness.run_evaluation")
    for module in (docker, constants, test_spec, harness):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    harness.log_root = tmp_path
    return client, harness


@pytest.fixture
def swebench_spec():
    return EvalSpec(
        task="SWEbench", source=SourceRef("fixture"), source_id="repo-1", ordinal=0,
        requests=(), grader=GraderSpec("swe_patch"), doc={"instance_id": "repo-1"},
    )


@pytest.mark.parametrize("resolved", [False, True])
def test_swebench_grades_and_closes_client_without_context_manager(swebench_boundary, swebench_spec, resolved):
    client, harness = swebench_boundary

    def run_instance(_instance, _prediction, **_kwargs):
        assert not client.closed
        return {"completed": True, "resolved": resolved}

    harness.run_instance = run_instance

    grade = swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert grade.metrics == {"accuracy": float(resolved)}
    assert client.closed


def test_swebench_harness_failure_closes_client_and_propagates(swebench_boundary, swebench_spec):
    client, harness = swebench_boundary

    def run_instance(_instance, _prediction, **_kwargs):
        raise RuntimeError("Docker daemon unavailable")

    harness.run_instance = run_instance

    with pytest.raises(RuntimeError, match="Docker daemon unavailable"):
        swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert client.closed


def test_swebench_invalid_model_patch_is_scored_zero(swebench_boundary, swebench_spec):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "run_instance.log").write_text("INFO - >>>>> Patch Apply Failed:\nNo valid patches in input\n")
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance

    grade = swe_patch(swebench_spec, ["This response is not a patch."])

    assert grade.metrics == {"accuracy": 0.0}
    assert client.closed


@pytest.mark.parametrize("failure", ["Error while pulling image", "Test timed out after 1800 seconds"])
def test_swebench_incomplete_infrastructure_is_not_scored_zero(swebench_boundary, swebench_spec, failure):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "run_instance.log").write_text(failure)
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance

    with pytest.raises(RuntimeError, match="did not complete"):
        swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert client.closed


def test_swebench_empty_model_patch_is_scored_zero(swebench_spec):
    grade = swe_patch(swebench_spec, [""])

    assert grade.metrics == {"accuracy": 0.0}
