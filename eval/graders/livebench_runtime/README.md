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
Keyword/letter frequencies, placeholders, highlighted sections, capital-word
counts, comma checks and language/case checks also use core comparisons, covering
all 25 current LiveBench and IFEval instruction IDs. Source builders, tokenizers
and language detectors prepare inputs; partial IFBench coverage is listed below. JSON-format answers with duplicate object keys, nonfinite numbers,
or excessive nesting score zero, even where the source parser accepts them.
Malformed trusted instruction arguments remain task errors rather than wrong
candidate answers. Language detection failures score zero. Case checks additionally reject uncased
alphabetic characters (such as CJK mixed with English), which source case checks
can accept. Cased nonalphabetic characters still must have the requested case.
The default source path is unchanged.

The shared instruction transport maps all 58 IFBench contracts to Schema, Exact,
IFEval and core aggregation. Source builders, tokenizers, taggers and emoji detection
provide task data; source `check_following` callbacks are never invoked by the
cutover. Unknown instruction IDs fail closed. Source evaluation remains the default;
set `verifyit_enabled=True` to select the cutover.

Integer-valued floating count metadata is normalized before comparison. Options
that normalize to empty labels, missing trigram references and nonfinite metadata
are invalid tasks. Blank responses score zero, including mixed batches. Trigram
precision uses literal character trigrams and an inclusive ±2 percentage interval;
case and whitespace remain significant. Empty references give nonempty candidates
zero precision. Prepared collections retain verifyit's 10,000-item/1,000,000-character
limits; oversized candidate counts score zero.

Calendar parsing rejects impossible dates, including zero month/day and nonleap
February 29. CSV grading validates every row even when an earlier row contains the
requested special character. Nested-bracket and quote grading require a fully closed group. Title-case checks
reject lowercase-leading mixed-case tokens such as `hELlo`.
These cases can receive credit on the source path and score zero on the cutover.
Indentation ignores blank lines consistently; the source's list mutation can reject
valid increasing indentation when consecutive blank lines are present. Finite prime word lengths through 97 remain part of the source task contract.

Distinct conjunctions are counted after lowercasing and stripping surrounding ASCII
punctuation. The equal-length sentence task also requires unique lowercase Unicode
word tokens (`\w+`), preventing repeated sentences from earning credit. Sub-bullets
require line-start `*` groups with a line-start `-` item in each group; inline marker
characters and prose without bullets do not satisfy this format.

AIW, AMC23 and GSM8K-Perturbed also accept `verifyit_enabled=True` in their benchmark
constructors (or `verifyit_enabled=true` in `--benchmark_args`). Install the same
`verifyit` extra shown above; these three routes need no additional runtime image.
AIW and AMC23 use the pinned Hendrycks answer normalization followed by strict Exact
comparison. GSM8K-Perturbed keeps its answer extraction and compares parsed numbers
through Numeric with zero tolerance. Missing candidate answers score zero. Invalid
trusted references raise `InvalidTask` before per-sample metrics are updated; source
scoring remains the default.

AIME24, AIME25 and MATH500 accept the same opt-in flag. Their cutover compares
extracted final expressions through Math and combines alternative references with
core MAX. Every trusted reference is checked before scoring. Strict box parsing
rejects incomplete boxes and unsupported percent or ordinal forms that the source
fallback may accept; the default source grader is unchanged.

Absent or nontext extracted Math candidates are treated as empty answers and score
zero. Trusted references are still validated first, so a missing candidate cannot
hide a malformed reference.

### Harness task opt-in

With the `verifyit` extra installed, `python -m eval.eval --verifyit_harness ...`
selects native verifyit grading for lm-eval-harness tasks. The Python entry points
`lm_eval.simple_evaluate` and `lm_eval.evaluator.evaluate` accept
`verifyit_enabled=True`. Enabled results record this choice in
`config.verifyit_enabled`; omitting the flag preserves source grading.

NQ-Open and TriviaQA retain their configured strict and extracted answer filters,
then use Exact and MAX for aliases. Their reserved invalid-extraction marker maps
to empty text, so a missing answer cannot receive credit against the alias
`invalid`; an explicit `Answer: invalid` remains a valid answer. Missing or empty
trusted alias lists invalidate the task. Answers that normalize to empty also
score zero, including punctuation-only answers the source scorer could credit.
TruthfulQA MC2 uses the MCQ primitive's
probability-mass policy; malformed or nonfinite likelihoods invalidate the run.

All 15 `uncheatable_eval_*` categories use the shared likelihood implementation
for corpus word/byte perplexity and bits per byte. Source word splitting and UTF-8
byte counts are preserved. These are unbounded diagnostics, not rewards; missing,
nonfinite, positive or overflowing log likelihoods abort reporting. The pinned
category preparation and metric configuration must match the supported source.

The `drop` override keeps its source short-answer extraction and normalizes spans
before core set-F1 and numeric gating. Its single predicted span is aligned with
the best gold span, retaining unmatched spans in the denominator and rounding
after aggregation. Missing or normalized-empty answers score zero, even where
the source could credit an empty normalized reference; malformed trusted spans
invalidate the task. Other response cardinalities are unsupported.

The flag rejects contracts without a native mapping instead of falling back to a
source scorer. The existing `gsm8k_verifyit` task remains a separate opt-in; use
it without `--verifyit_harness`.
