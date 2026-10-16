"""One patch graded by SWE-bench's single-instance Docker harness."""

from contextlib import closing

from eval.contracts.eval_spec import Grade
from eval.runtime.checkpoints import input_digest


def swe_patch(spec, responses):
    import docker
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    patch = str(responses[-1])
    instance = make_test_spec(dict(spec.doc), namespace="swebench", instance_image_tag="v1")
    prediction = {KEY_INSTANCE_ID: spec.doc["instance_id"], KEY_MODEL: "evalchemy", KEY_PREDICTION: patch}
    with closing(docker.from_env()) as client:
        outcome = run_instance(instance, prediction, rm_image=False, force_rebuild=False, client=client,
                               run_id="evalchemy-" + input_digest(prediction)[:16], timeout=1800)
    if not outcome["completed"]:
        raise RuntimeError(f"SWE-bench did not complete instance {spec.source_id}")
    return Grade({"accuracy": float(outcome["resolved"])})
