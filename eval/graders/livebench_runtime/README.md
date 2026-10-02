# LiveBench source runtime

Build from the Evalchemy repository root:

```sh
docker build -t verifyit-evalchemy-livebench:source-v1 -f eval/graders/livebench_runtime/Dockerfile .
```

The base Python image and all Python runtime packages are pinned. The PyExt fork
commit matches the source evaluation environment and supports Python 3.12; the
source contract module requires Python 3.11 or later. `VERIFYIT_LIVEBENCH_IMAGE`
can select another prebuilt image with the same dependencies.

The default LiveBench path remains unchanged. Enable the cutover explicitly with
`LiveBenchBenchmark(verifyit_enabled=True, ...)` or the equivalent benchmark
configuration. The opt-in client requires verifyit with its schema extra and the
companion harness modules `lm_eval.verifyit_humaneval` and
`lm_eval.verifyit_function_worker`; an upstream-only harness installation does
not provide these modules. Client dependency publication is separate from the
verifyit package release.

Add `eval/chat_benchmarks/LiveBench` to `PYTHONPATH` alongside the Evalchemy root:
source preprocessing imports the vendored `livebench` package by its top-level
name. Host callbacks retain the source optional dependencies, including
Levenshtein for branches that still use their source grading contract.

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
and core aggregation; other retained branches remain documented coverage gaps.

The source callback manifest guards the vendored implementation and shared code
it imports. Instruction detection is deterministically seeded, and an undetectable
language receives zero instead of the source's success fallback. Empty or malformed
trusted contracts abort with `invalid_task`; ordinary malformed candidate answers
receive zero. An empty eligible task cannot reuse an earlier judgment file.
