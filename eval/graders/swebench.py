"""One patch graded by SWE-bench's single-instance Docker harness."""

from contextlib import closing

from eval.contracts.eval_spec import Grade
from eval.runtime.checkpoints import input_digest


def swe_patch(spec, responses):
    patch = str(responses[-1])
    if not patch:
        return Grade({"accuracy": 0.0})

    import docker
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_INSTANCE, RUN_EVALUATION_LOG_DIR,
    )
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance = make_test_spec(dict(spec.doc), namespace="swebench", instance_image_tag="v1")
    prediction = {KEY_INSTANCE_ID: spec.doc["instance_id"], KEY_MODEL: "evalchemy", KEY_PREDICTION: patch}
    run_id = "evalchemy-" + input_digest(prediction)[:16]
    with closing(docker.from_env()) as client:
        outcome = run_instance(instance, prediction, rm_image=False, force_rebuild=False, client=client,
                               run_id=run_id, timeout=1800)
    if not outcome["completed"]:
        # The harness uses the same incomplete result for rejected model patches
        # and infrastructure errors; its patch-apply marker identifies a scored failure.
        log_path = RUN_EVALUATION_LOG_DIR / run_id / prediction[KEY_MODEL] / prediction[KEY_INSTANCE_ID] / LOG_INSTANCE
        if APPLY_PATCH_FAIL in log_path.read_text():
            return Grade({"accuracy": 0.0})
        raise RuntimeError(f"SWE-bench did not complete instance {spec.source_id}; see {log_path}")
    return Grade({"accuracy": float(outcome["resolved"])})
