# Plan — harbor-style retry + error classification for evalchemy's API-eval path

**Status:** PROPOSAL for review. No implementation until greenlit.
**Scope:** the `eval/robust_api.py` monkeypatch layer (merged in #18) over lm-eval's
`TemplateAPI.get_batched_requests` / `amodel_call` (`local-completions` /
`local-chat-completions`, i.e. every `eval.serve_eval` endpoint eval).
**Author branch:** `feuer/retry-error-classification-plan` (fresh; no self-merge).

---

## 1. Problem

Today evalchemy's API path retries (or fails) requests **without asking why they failed**.
The mechanics inherited from lm-eval v0.4.12 (`lm_eval/models/api_models.py`) are:

- `get_batched_requests` wraps `amodel_call` in a single tenacity policy:
  `retry(stop=stop_after_attempt(self.max_retries), wait=wait_exponential(0.5, 1, 10), reraise=True)`
  — applied **uniformly to every exception**.
- `#18`'s `eval/robust_api.py` fixed the batch-abort (a request that exhausts retries now
  scores as a MISS instead of nuking the whole `gather`), but it still treats **all**
  failures identically: same retry budget, same backoff, same "→ miss" outcome.

Two costs of the no-classification approach:

1. **Wasted budget on non-convergent failures.** A prompt that overflows the model's
   context window (HTTP 400 "maximum context length") will fail *identically* on every
   retry — retrying it burns wall-clock and (on a metered endpoint) money, and never
   converges. Same for a persistently truncated generation.
2. **Systemic faults masked as a plausible score.** An endpoint-wide fault — 401 auth,
   a malformed harness payload (400), the server being down — hits *every* request. The
   #18 patch converts all of them to misses, so the run **completes with a bogus ~0.0
   score that looks real**. This is the same class of bug the original UnboundLocalError
   *masking* had: the operator can't tell "the model scored 0" from "the run was broken."

The operator's explicit ask: make the **infra-vs-model** distinction the centerpiece, and
add **politeness** (jittered backoff, honor `Retry-After`, don't hammer a struggling server).

---

## 2. What harbor does (evidence)

harbor already runs exactly this discipline. Mined from the local clone
(`marin-community/harbor @ penfever/working`):

### 2.1 Classification-driven retry — `src/harbor/llms/lite_llm.py`
The `LiteLLM.call()` retry decorator (lines 622-638):
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=( retry_if_exception_type(Exception)
          & retry_if_not_exception_type(
              (ContextLengthExceededError, OutputLengthExceededError, openai.AuthenticationError)) ),
    reraise=True,
)
```
**Retry everything transient; NEVER retry the three passthrough classes** —
context-overflow, output-truncation, auth. These are raised as *typed harbor exceptions*
by `_handle_llm_error` (lines 1083-1099) + the `finish_reason == "length"` check
(lines 873-880), translated from raw `openai` errors:
- `AuthenticationError` → re-raise (fatal config).
- `BadRequestError` / `APIStatusError` **with a context-length signal** →
  `ContextLengthExceededError`. The signal is a string match over the error body
  (`_is_context_length_error`, lines 1117-1137: `"maximum context length"`,
  `"context_length_exceeded"`, `"prompt is too long"`, `"model's context length"`, …).
- `finish_reason == "length"` → `OutputLengthExceededError(truncated_response=...)`.

harbor also **disables the SDK's own retries** (`AsyncOpenAI(max_retries=0)`,
`_get_openai_client`, line 431) precisely so *harbor* owns retry and the transport can't
double-retry underneath it. And it clamps a context-exhausted `max_tokens` up to `>=1`
(lines 719-730) so a degenerate budget returns a clean short completion instead of a 400.

### 2.2 Per-class retry *budgets* + *wait* — `src/harbor/environments/daytona/utils.py`
`daytona_retry_callbacks` (lines 227-268) is the richer template — different budgets AND
different backoff **per class**, via tenacity retry/wait *callbacks*:
```python
def retry_callback(state):
    exc = state.outcome.exception()
    if exc is None or _is_non_retryable(exc):     return False                       # fail fast
    if is_transient_daytona_error(exc):           return state.attempt_number < 10   # capacity: patient
    if is_daytona_auth_flake(exc):                return state.attempt_number < 6    # bounded
    return state.attempt_number < 3                                                  # default

def wait_callback(state):
    if is_transient_daytona_error(exc):  return transient_linear_step * state.attempt_number  # linear 60,120,180…
    return min(30, max(2, 2**state.attempt_number))                                           # bounded exp
```
`is_transient_daytona_error` (line ~130) classifies by exception TYPE
(`DaytonaRateLimitError`) **and** by string patterns
(`_TRANSIENT_MESSAGE_PATTERNS = {"limit exceeded", "too many requests", "capacity", "rate limit"}`).

### 2.3 Politeness / jitter — the adapters
harbor's main LLM path has **no** jitter and does **not** honor `Retry-After` (a gap we
will close), but its adapters show the intended shape:
- `adapters/scienceagentbench/llm_visual_judge.py` (429 handler): `OpenAI(max_retries=0)`,
  `wait = 60`, `jitter = random.uniform(0, wait*0.1)`, `sleep(wait+jitter)`,
  `wait = min(wait*2, 480)` — long, jittered, capped backoff dedicated to 429.
- `adapters/financeagent/.../google_search.py`: `backoff.full_jitter` +
  `giveup=lambda e: not is_429(e)`.

### 2.4 Trial-level "benign vs not-scored" — `src/harbor/trial/errors.py`
`AgentTimeoutError` subclasses `asyncio.TimeoutError` and *deliberately does NOT* subclass
`TrialNotScoredError` — a timeout is **benign**: the trial IS scored (as a failure).
`VerificationNotCompletedError` IS `TrialNotScoredError` → the trial is unscored → prune +
re-run. This is the exact "score-as-failure vs re-run vs fail-loud" three-way we want at
the *request* level, and it aligns with OT-Agent's harvest gate (`scripts/database/
eval_guardrail.py`, the `crud-purge-below-gate-evals` benign-set) — so request-level
classes should roll up into the same benign/non-benign accounting.

---

## 3. Proposed taxonomy for evalchemy

The lm-eval endpoint path uses **raw `aiohttp`** (not the openai SDK): `amodel_call` does
`session.post(...)` → `response.raise_for_status()` → `response.json()`. So the exceptions
we classify are `aiohttp` errors (carrying `.status` and `.headers`), `asyncio`
timeouts, and body-parse errors — plus one *non-exception* signal (`finish_reason`).

| # | Class | Trigger (what arrives at the retry seam) | Retry policy | Terminal outcome (generate) |
|---|-------|------------------------------------------|--------------|------------------------------|
| 1 | **INFRA_TRANSIENT** | `ClientConnectionError`/`ClientOSError` (conn reset), `asyncio.TimeoutError`/`ServerTimeoutError`, HTTP **5xx**, body `JSONDecodeError`/`ContentTypeError` (garbled/partial body) | RETRY, exp backoff **+ full jitter** | after budget → MISS (benign) |
| 2 | **RATE_LIMIT** | HTTP **429** (`ClientResponseError.status == 429`) | RETRY, **honor `Retry-After`**, longer jittered backoff, optional global cooldown | after budget → MISS (benign) |
| 3 | **MODEL_CONTEXT** | HTTP **400** whose body matches a context-length phrase (harbor's `_is_context_length_error` set) | **NO retry** (never converges) | immediate MISS (model-driven) |
| 4 | **MODEL_TRUNCATION** | 200 response with `finish_reason == "length"` (not an exception) | **NO retry** | grade the (partial) text as-is — already a miss; **count it** |
| 5 | **FATAL_CONFIG** | HTTP **401/403** (auth), or **400** that is NOT context-length (bad payload / unrecognized arg) | **NO retry** | **RE-RAISE → abort the run loud** (surface the real cause) |
| 6 | **UNKNOWN** | anything unmatched | RETRY (default, like harbor "retry unless proven passthrough") | after budget → MISS (benign) |

**The load-bearing design decisions:**

- **Infra (1,2,6) → retry then degrade to a miss.** Transient faults on a healthy server
  shouldn't lose the run; if they persist past budget, one miss is better than an abort.
- **Model-driven (3,4) → no retry, score as failure.** A too-long prompt or a truncated
  generation is a real *model* failure on *that item*; retrying is pure waste. Scoring it
  as a miss is the correct benchmark semantics (the model didn't produce a right answer).
- **Fatal (5) → fail loud, do NOT mask as misses.** This is the crucial refinement over
  #18: an auth error or malformed payload hits *every* request, so converting them to N
  misses fabricates a fake 0.0 score. Re-raising surfaces the real cause — exactly the
  masking failure mode the operator has hit twice. (A run with a few genuine per-item
  misses is fine; a run where the *endpoint contract* is broken must not look "done".)

This keeps #18's core guarantee (one bad *item* never nukes the batch) while adding: don't
waste budget on non-convergent items, and don't hide a broken *run*.

---

## 4. Where each hook lands

All changes stay in **`eval/robust_api.py`** (monkeypatch; no lm-eval fork, no upstream
`eval/` edit). Three additions, plus a rework of the retry construction already in the
patched `get_batched_requests`:

### 4.1 `classify_api_error(exc) -> ErrorClass` (new, pure function)
Inspect the exception the same way harbor's `is_transient_daytona_error` /
`_is_context_length_error` do:
```
if ClientResponseError:
    s = exc.status
    if s == 429:                              -> RATE_LIMIT
    if 500 <= s < 600:                        -> INFRA_TRANSIENT
    if s in (401, 403):                       -> FATAL_CONFIG
    if s == 400:
        -> MODEL_CONTEXT if _is_context_length(exc) else FATAL_CONFIG
if (ServerTimeoutError, asyncio.TimeoutError, ClientConnectionError, ClientOSError):
                                              -> INFRA_TRANSIENT
if (JSONDecodeError, ContentTypeError):       -> INFRA_TRANSIENT
else:                                         -> UNKNOWN
```
`_is_context_length(exc)` ports harbor's phrase set verbatim (string-match `str(exc)` +
`exc.message` + `exc.headers`-adjacent body). This function is trivially unit-testable in
isolation — no network, no model.

> **Dependency note (dependency-ground-truth-uv):** classification reads only symbols
> already present via `lm-eval[api]` (aiohttp, stdlib `asyncio`/`json`). **No pyproject/
> uv.lock change.** If we later want the openai SDK's typed errors we would revisit, but
> the raw-aiohttp path needs nothing new.

### 4.2 Per-class retry/wait callbacks in the patched `get_batched_requests`
Replace the single `retry(...)` currently built inside `get_batched_requests` with
tenacity **callbacks** (the `daytona_retry_callbacks` shape):
```python
def _retry(state):
    exc = state.outcome.exception()
    cls = classify_api_error(exc)
    if cls in (MODEL_CONTEXT, MODEL_TRUNCATION, FATAL_CONFIG):  return False   # never retry
    return state.attempt_number < _BUDGET[cls]                                  # INFRA/RATE_LIMIT/UNKNOWN

def _wait(state):
    return _backoff_for(classify_api_error(state.outcome.exception()), state)   # §5
retry_ = retry(retry=_retry, wait=_wait, reraise=True)(self.amodel_call)
```
The existing `_guarded` wrapper stays, but its terminal `except` becomes class-aware:
```python
except BaseException as exc:
    if not generate:                 raise                    # loglikelihood unchanged (fail-fast)
    cls = classify_api_error(exc)
    if cls == FATAL_CONFIG:          raise                    # NEW: surface systemic faults loud
    _counts[cls] += 1                                         # instrument
    ...log with cls..., cache misses, return [placeholder]*n  # benign/model → miss (as #18)
```
Net: #18's "single item → miss" is preserved for classes 1-4/6; only **FATAL_CONFIG**
newly re-raises (correctly aborting a broken run).

### 4.3 `MODEL_TRUNCATION` counting (Phase 2, optional)
`finish_reason` is visible inside `amodel_call` (in the parsed `outputs`), not at the
`get_batched_requests` seam. To count truncations we wrap `parse_generations` (or add a
light post-parse hook) to increment `_counts[MODEL_TRUNCATION]` when a choice's
`finish_reason == "length"`. **Scoring is unchanged** (the truncated text already grades as
wrong); this is visibility only. Defer to Phase 2 to keep the MVP on the exception path.

### 4.4 Run summary (instrumentation)
At the end of `get_batched_requests` (or via an atexit/`__del__` on the model), log one
line: `robust_api: infra_retried=N rate_limited=N context_overflow=N(miss) truncated=N
fatal=N(abort)`. Feeds the operator's "why did it fail" question and the harvest gate's
benign-vs-real accounting. Cheap, high-value.

---

## 5. Backoff + politeness design (concrete defaults)

All tunable via env (read once at import, like lm-eval's `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER`):

| Class | Max attempts | Backoff `wait(n)` | Notes |
|-------|-------------:|-------------------|-------|
| INFRA_TRANSIENT / UNKNOWN | **5** (`EVALCHEMY_RETRY_MAX_INFRA`) | `full_jitter(min(30, 1·2^(n-1)))` → ~0-1,0-2,0-4,0-8,0-16,cap 30 | jitter = `uniform(0, computed)` |
| RATE_LIMIT (429) | **8** (`EVALCHEMY_RETRY_MAX_429`) | `Retry-After` if present, else `full_jitter(min(120, 5·2^(n-1)))` | see below |
| MODEL_CONTEXT / MODEL_TRUNCATION | 1 (no retry) | — | immediate miss |
| FATAL_CONFIG | 1 (no retry) | — | re-raise |

Knobs: `EVALCHEMY_RETRY_BASE_S=1`, `EVALCHEMY_RETRY_CAP_S=30`, `EVALCHEMY_RETRY_429_BASE_S=5`,
`EVALCHEMY_RETRY_429_CAP_S=120`, `EVALCHEMY_RETRY_JITTER=full` (`full`|`none`).

**Politeness specifics:**
1. **Full jitter** (`uniform(0, backoff)`, AWS-style) on every retryable class — de-syncs
   the `num_concurrent` coroutines so they don't retry in lockstep and re-spike the server.
2. **Honor `Retry-After`** on 429: parse `exc.headers.get("Retry-After")` as either an int
   (seconds) or an HTTP-date (`email.utils.parsedate_to_datetime`); clamp to
   `[1, EVALCHEMY_RETRY_429_CAP_S]`; if absent, use the jittered 429 backoff.
3. **Adaptive global cooldown (Phase 2, optional):** on any 429, set a process-shared
   "not-before" timestamp = `now + wait`; every coroutine's `_wait` returns
   `max(class_wait, not_before - now)`. This makes *one* 429 briefly slow *all* in-flight
   requests — the strongest "don't hammer a struggling server" measure. Gate behind
   `EVALCHEMY_RETRY_ADAPTIVE=1` (default off) until validated.
4. **No double-retry.** lm-eval's aiohttp path has no SDK auto-retry to disable (unlike
   harbor's `AsyncOpenAI(max_retries=0)`); we note this so a future openai-SDK migration
   remembers to set `max_retries=0`.
5. **Concurrency cap on retries.** Retries already run under the existing
   `asyncio.Semaphore(self._concurrent)` (acquired inside `amodel_call`), so in-flight
   retries are already bounded to `num_concurrent`. No new semaphore needed for the MVP;
   the adaptive cooldown (3) is the escalation if that proves insufficient.

Defaults chosen to be **safe/patient** (rate limits get 8 tries over up to ~2 min; infra
gets 5 over ~30 s) and to **never** retry a non-convergent class.

---

## 6. Test plan

**Unit — classifier (`classify_api_error`), no network:**
- Synthetic `aiohttp.ClientResponseError` with `status` ∈ {429, 500, 502, 400+ctx-body,
  400+other, 401, 403}; `asyncio.TimeoutError`, `ServerTimeoutError`,
  `ClientConnectionError`, `json.JSONDecodeError`, a bare `RuntimeError` → assert the
  expected class for each. Port harbor's `_is_context_length_error` phrase fixtures.

**Unit — retry/outcome (fake `TemplateAPI` stub, the #18 test harness extended):**
- `amodel_call` raising each class → assert:
  - INFRA_TRANSIENT/RATE_LIMIT/UNKNOWN: retried up to the budget, then one MISS returned
    (batch not aborted); attempt count == budget.
  - MODEL_CONTEXT: **attempt count == 1** (no retry), MISS returned.
  - FATAL_CONFIG: **re-raised** (no miss), attempt count == 1.
  - loglikelihood (`generate=False`): every class re-raises (unchanged from #18).
- Assert misses are cache-written (`cache_hook.add_partial`) so `--use_cache` resume works.

**Unit — backoff/politeness:**
- `_wait` returns values within `[0, cap]` per class; INFRA grows ~exponentially; jitter
  makes repeated calls non-identical.
- 429 with a `Retry-After: 7` header → wait ≈ 7 (clamped); with an HTTP-date header →
  parsed correctly; with no header → falls back to the jittered 429 curve.

**Integration — the local mlx smoke (already stood up in #18):**
- A tiny fault-injecting reverse proxy in front of the served endpoint:
  - 20% → 503 → assert the gsm8k run **completes with a score**; summary shows
    `infra_retried>0`; score ≈ the fault-free baseline (retries absorbed the faults).
  - a few prompts → 400 context-length → assert those score as misses, run completes,
    summary shows `context_overflow>0`, and NO retries were spent on them.
  - all → 401 → assert the run **aborts loudly** with the auth cause (does NOT report a
    fake 0.0). This is the anti-masking regression test.

**Regression — flag-off / healthy parity:**
- With every request healthy, assert **byte-identical** behavior to the current #18 patch:
  no extra retries, identical gsm8k/MATH500 scores on the mlx smoke (the numbers from the
  #18 validation: gsm8k flex 0.1333, MATH500 0.03). Classification must be a no-op when
  nothing fails.

---

## 7. Scope, phasing, risks

- **Phase 1 (MVP, the operator's core ask):** §4.1 classifier + §4.2 per-class retry/
  outcome + §4.4 summary + §5 (1,2,4,5) politeness. Entirely within `eval/robust_api.py`.
- **Phase 2 (optional):** §4.3 truncation counting, §5.3 adaptive global cooldown.
- **Non-goals:** no lm-eval fork; no pyproject/uv.lock change; no change to loglikelihood
  semantics; no change to how a *truncated but successful* generation is *scored* (only
  counted). The image (`:evalchemy-gpu`) rebuilds unchanged — `uv sync --frozen` off the
  same lock; only `eval/robust_api.py` differs, and the build-time assert already proves it
  loads.
- **Risk — classifier drift:** lm-eval's aiohttp error shapes could change across
  versions; the classifier is defensive (unknown → retry, the safe default) and pinned to
  v0.4.12 like the rest of #18. Covered by the classifier unit tests.
- **Risk — FATAL re-raise is a behavior change vs #18** (a 401/400-run now aborts instead
  of "completing" with 0.0). This is *intended* (anti-masking) but is the one place a
  reviewer should confirm the desired default; it can be gated behind
  `EVALCHEMY_FATAL_ABORT=1` (default on) if you want an escape hatch.

## 8. Open questions for review
1. **FATAL_CONFIG default** — abort-loud (proposed) vs still-degrade-to-miss-but-flag? I
   recommend abort; confirm.
2. **429 budget/caps** — 8 attempts / 120 s cap assumes a self-hosted vLLM that rarely
   429s and a metered endpoint that does. Acceptable, or want it lower for self-hosted?
3. **Adaptive cooldown** — worth Phase-1, or leave Phase-2/off until a real 429 storm?
4. **Truncation** — count only (proposed), or also optionally single-retry with a larger
   `max_gen_toks` when the task budget allows? (I lean count-only: the task owns the budget.)
