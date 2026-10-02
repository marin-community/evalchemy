# LiveBench source runtime

Install the pinned client and runtime dependencies from the repository root:

```sh
uv sync --locked --extra livebench --extra verifyit --no-dev --no-editable
docker build -t verifyit-evalchemy-livebench:source-v1 -f eval/graders/livebench_runtime/Dockerfile .
```

The Docker recipe pins the base image, Python packages and PyExt fork. Docker is
required for coding and AMPS. `VERIFYIT_LIVEBENCH_IMAGE` can select a prebuilt image
with the same dependencies. Source hashes in `source-hashes.json` reject unsupported
vendored grader revisions.

Use the task loader with the same dataset and release used during generation:

```python
import json
from pathlib import Path
from eval.task import TaskManager

manager = TaskManager(
    task_list=["LiveBench"], dataset_name="live_bench", release_date="2024-08-31",
    verifyit_enabled=True,
)
benchmark = manager.get_benchmark("LiveBench")
if benchmark is None:
    raise RuntimeError(manager.load_failures)
responses = [json.loads(line) for line in Path("responses.jsonl").read_text().splitlines() if line.strip()]
results = benchmark.evaluate_responses(responses)
```

Response rows retain source `question_id`, `model_id` and `choices` fields.
Omitting `verifyit_enabled` preserves source grading. The enabled module registers
the vendored `livebench` package and rejects a conflicting namespace.

Coding candidates run without network access in a container with a read-only
root, 1 GiB memory, one CPU and a 64-process limit. Candidate source and inputs
travel through stdin; the only mount is the read-only RPC worker. Trusted
references and grading remain outside the container. Calls have six-second
deadlines; the outer Script deadline is 120 seconds including startup. Both
supervisors remove the exact named container on failure.

JSONSchema compares functional outputs, StdIO compares decimal lines and ALL
combines tests. Source singleton-list and approximate floating-point fallbacks
are unsupported. Invalid trusted references abort; candidate exceptions, broken
transport, malformed output and timeouts score zero. AMPS uses its source runtime.
Reasoning, tables and retained-answer routes prepare observations for existing
core primitives. Proof rearrangement, house traversal and plot unscrambling
retain their declared source scorers; unknown routes fail closed.

Instruction-following uses the shared IFEval/IFBench transport and named source
preparation. Unknown IDs fail closed. Source tokenizers, taggers and emoji tools
provide observations; enabled routes do not call source `check_following` graders.
Preparation preserves the declared instruction collection limits, finite count
admission and blank-response policy.

Other opt-ins are documented in the [main README](../../../README.md) and
[Judge guide](../verifyit_judges.md). The `--verifyit_harness` flag selects the
pinned companion harness's native mappings; unsupported contracts fail closed.
HumanEval harness grading requires `--confirm_run_unsafe_code`, one completion,
`pass@1` and its pinned candidate image:

```sh
docker pull python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e
```
