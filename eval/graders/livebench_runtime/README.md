# LiveBench source runtime

Build from the Evalchemy repository root:

```sh
docker build -t verifyit-evalchemy-livebench:source-v1 -f eval/graders/livebench_runtime/Dockerfile .
```

The base Python image and all Python runtime packages are pinned. The PyExt fork
commit matches the source evaluation environment and supports Python 3.12; the
source contract module requires Python 3.11 or later. `VERIFYIT_LIVEBENCH_IMAGE`
can select another prebuilt image with the same dependencies.

Install the pinned client dependencies from the Evalchemy checkout:

```sh
uv sync --locked --extra livebench --extra verifyit --no-dev --no-editable
```

The lock pins verifyit and the companion harness by immutable Git commit. The
harness companion has the same upstream v0.4.12 base as the default evaluator and
adds the isolated candidate worker. No `PYTHONPATH` setting is needed. Docker is
required for coding and AMPS; build the image above before evaluating those tasks.

The source grader remains the default. Use the normal task loader to enable the
cutover for an existing LiveBench response JSONL:

```python
import json
from pathlib import Path
from eval.task import TaskManager

manager = TaskManager(
    task_list=["LiveBench"],
    dataset_name="live_bench",
    release_date="2024-08-31",
    verifyit_enabled=True,
)
benchmark = manager.get_benchmark("LiveBench")
if benchmark is None:
    raise RuntimeError(manager.load_failures)
responses = [json.loads(line) for line in Path("responses.jsonl").read_text().splitlines() if line.strip()]
results = benchmark.evaluate_responses(responses)
```

Run this with `.venv/bin/python`; response rows retain the source `question_id`,
`model_id`, and `choices` fields. Use the same dataset and release as generation.
Omit `verifyit_enabled` or set it to false to use the source grader. The task
loader resolves vendored source imports; bounded workers validate and register
the same installed package, rejecting a conflicting `livebench` namespace.

Coding uses this image as an isolated candidate worker. Its only bind mount is
the read-only RPC worker; candidate source and inputs travel through stdin.
Trusted references and core grading remain outside the container. Containers
have no network, a read-only root filesystem, a 1 GiB memory limit, one CPU and
64-process limit. Each call has a six-second deadline; the outer Script deadline
is 120 seconds including startup. Both the shared worker context and outer
supervisor clean up the exact named container on failure.

Core JSONSchema compares functional outputs, StdIO compares decimal lines, and
ALL combines test outcomes. Source singleton-list and approximate floating-point
fallbacks are not used. Undefined/nonfinite references abort; candidate exceptions,
malformed output and timeouts receive zero. AMPS remains in the source runtime.
Zebra, web-of-lies and spatial routes prepare answers for core Exact/JSONSchema
and core aggregation. Table reformatting uses core JSONSchema for structure and
nulls, Numeric for finite numeric tolerance, and ALL/MAX for results. Table joins
prepare canonical key/value labels for Exact set-overlap F1, rounded to two decimal
places. Empty candidate values or nonstring labels conservatively score zero; empty or
malformed reference mappings are invalid tasks. Other retained branches remain
documented coverage gaps.

The source callback manifest guards the vendored implementation and shared code
it imports. Instruction detection is deterministically seeded, and an undetectable
language receives zero instead of the source's success fallback. Empty or malformed
trusted contracts abort with `invalid_task`; ordinary malformed candidate answers
receive zero. An empty eligible task cannot reuse an earlier judgment file.

For instruction following, the cutover delegates keyword presence/absence,
word/sentence/paragraph counts, paragraph first words, bullet/section counts,
constrained responses, titles, JSON format, postscripts, quotation/end checks,
and repeated-prompt/two-response checks to existing Schema or IFEval grading.
Source builders and tokenizers still prepare their inputs. Other instruction
IDs retain their source predicate path; this is not complete instruction-family
migration. JSON-format answers with duplicate object keys, nonfinite numbers,
or excessive nesting score zero, even where the source parser accepts them.
Malformed trusted instruction arguments remain task errors rather than wrong
candidate answers. The default source path is unchanged.
