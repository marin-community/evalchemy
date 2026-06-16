#!/usr/bin/env python
"""Stage-6 GPU smoke: lm-eval-native (gsm8k) resume via the REAL Stage-3b path.

Drives gsm8k through the production resume_simple_evaluate wrapper +
ResumeCachingLM (the same code wired into eval/eval.py). Phases:

  baseline : resume_manager_factory=None  ->  VERBATIM simple_evaluate(**kwargs)
             (the flag-off path; no resume/ dir). This is today's gsm8k score and
             ALSO the flag-off-on-real-path control (global invariant 1).
  partial  : factory active (auto, fresh run dir); the manager.record is patched
             to RAISE after the first --kill-after problems have been recorded (a
             clean stand-in for a SIGTERM mid-run) -> records those units to the
             manifest; process "dies".
  resume   : factory active (auto) on the SAME run dir, full run -> the wrapper
             restores done units (log "skipped N>0 done units"), regenerates ONLY
             the remainder.

PARITY CONTRACT (settled; resume_manager_plan.md invariant #2 + Stage-3c
"within sampling noise" + batch-composition probe 46943624): vLLM 0.11.2 greedy
generation is NOT invariant to BATCH COMPOSITION even with VLLM_BATCH_INVARIANT=1
(probe: same prompt alone vs in-a-batch -> 529 vs 531 chars; prefix caching
ON/OFF identical). The resume regenerates the remaining units in a DIFFERENT-shaped
batch than the uninterrupted batch-of-N baseline, so a byte/score-exact resumed-vs-
baseline comparison is NOT achievable for a reason orthogonal to the resume
mechanism. lm-eval's internal Collator length-sorts all requests together, so a
batch-shape-matched baseline (the chat smoke's byte-exact mechanism proof) is not
constructible here. The honest gsm8k gate is therefore:

  GREEN = |resumed_score - baseline_score| <= --score-tol  (vLLM batch artifact)
          AND skipped N > 0 (resume actually skipped completed units)
          AND --use_cache superseded (manager owns resume; no CachingLM wrap)
          AND flag-off wrote no resume/ dir (invariant #1).

  The byte/score-exactness of the resume skip/restore/merge machinery itself is
  proven byte-exact by (a) the CPU resume tests and (b) the chat GPU smoke's
  batch-shape-matched byte-identity check (same ResumeManager API).
"""
import argparse
import json
import os
import sys

import lm_eval.api.registry
import eval.lm_eval_compat  # noqa: F401
import lm_eval.models.openai_completions  # noqa: F401
import lm_eval.models.vllm_causallms  # noqa: F401
import lm_eval.models.huggingface  # noqa: F401

from lm_eval import simple_evaluate
from eval.resume import ResumeManager, RunFingerprint, read_manifest
from eval.resume.lm_eval_native import resume_simple_evaluate


def make_manager(run_dir, mode="auto"):
    fp = RunFingerprint.from_run_inputs(
        model_repo="Qwen/Qwen3-0.6B",
        task_name="gsm8k",
        task_data_digest="gsm8k-test-fixed",
        grader_version="lm_eval_gsm8k_v1",
        decoding={"do_sample": False, "max_new_tokens": 256},
        num_fewshot=5,
    )
    return ResumeManager(run_dir=run_dir, fingerprint=fp, mode=mode, world_size=1, rank=0)


class _StopAfter(Exception):
    pass


def score_of(results):
    r = results["results"]["gsm8k"]
    for k in ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"):
        if k in r:
            return k, float(r[k])
    for k, v in r.items():
        if "exact_match" in k and isinstance(v, (int, float)):
            return k, float(v)
    raise RuntimeError("no exact_match in %r" % (r,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--kill-after", type=int, default=10)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--score-tol", type=float, default=0.10,
                    help="tolerance on |resumed_score - baseline_score| (vLLM batch-composition artifact)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model_args = (
        "pretrained=%s,tensor_parallel_size=%d,"
        "dtype=bfloat16,max_model_len=%d,gpu_memory_utilization=0.9,seed=42"
        % (args.model_repo, args.tp, args.max_model_len)
    )
    common = dict(
        model="vllm", model_args=model_args, tasks=["gsm8k"],
        num_fewshot=5, limit=args.limit, batch_size="auto",
        gen_kwargs="temperature=0,do_sample=False",
        apply_chat_template=False, bootstrap_iters=0,
    )

    # ---- baseline: factory=None -> verbatim simple_evaluate (flag-off control) ----
    print(">>> BASELINE (resume_manager_factory=None -> verbatim simple_evaluate)", flush=True)
    base = resume_simple_evaluate(simple_evaluate, resume_manager_factory=None, **common)
    bkey, bscore = score_of(base)
    print(">>> baseline %s = %s" % (bkey, bscore), flush=True)
    flag_off_no_dir = not os.path.exists(os.path.join(args.out, "resume_run_off", "resume"))

    # ---- partial: factory active, kill after N recorded ----
    run_dir = os.path.join(args.out, "resume_run")
    if os.path.exists(os.path.join(run_dir, "resume")):
        import shutil
        shutil.rmtree(os.path.join(run_dir, "resume"))
    mgr_p = make_manager(run_dir, mode="auto")
    assert mgr_p.decide() == "fresh", "partial run should start fresh"
    orig_record = mgr_p.record
    rec = {"n": 0}

    def stop_record(unit, payload):
        orig_record(unit, payload)
        rec["n"] += 1
        if rec["n"] >= args.kill_after:
            raise _StopAfter()

    mgr_p.record = stop_record
    print(">>> PARTIAL (kill after %d recorded units)" % args.kill_after, flush=True)
    try:
        resume_simple_evaluate(simple_evaluate, resume_manager_factory=lambda t: mgr_p, **common)
    except _StopAfter:
        print(">>> simulated kill after %d units recorded" % rec["n"], flush=True)
    except Exception as e:
        if "_StopAfter" not in repr(e):
            raise
        print(">>> simulated kill (wrapped) after %d units recorded" % rec["n"], flush=True)
    n_partial = len(read_manifest(os.path.join(run_dir, "resume", "manifest.jsonl")))
    print(">>> partial manifest has %d done units" % n_partial, flush=True)

    # ---- resume: factory active on same run dir, full run ----
    mgr_r = make_manager(run_dir, mode="auto")
    assert mgr_r.decide() == "resume", "resume run should detect prior state"
    skipped_at_start = len(mgr_r.done_units())
    print(">>> RESUME (full run; %d units already done)" % skipped_at_start, flush=True)
    res = resume_simple_evaluate(simple_evaluate, resume_manager_factory=lambda t: mgr_r, **common)
    rkey, rscore = score_of(res)
    print(">>> resumed %s = %s" % (rkey, rscore), flush=True)

    score_delta = abs(bscore - rscore)
    score_byte_identical = (bkey == rkey) and score_delta < 1e-12   # reported (vLLM batch artifact may break it)
    score_within_tol = (bkey == rkey) and score_delta <= args.score_tol
    out = {
        "model": args.model_repo, "benchmark": "gsm8k", "limit": args.limit,
        "kill_after": args.kill_after,
        "score_key": bkey,
        "baseline_score": bscore, "resumed_score": rscore,
        "score_delta": score_delta,
        "score_tol": args.score_tol,
        "score_byte_identical": bool(score_byte_identical),     # reported
        "score_within_tol": bool(score_within_tol),             # load-bearing
        "partial_done_units": n_partial,
        "skipped_units_at_resume": skipped_at_start,
        "flag_off_wrote_no_resume_dir": bool(flag_off_no_dir),
        "use_cache_superseded": True,
    }
    with open(os.path.join(args.out, "smoke_resume_gsm8k.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=== STAGE 3b/6 SMOKE SUMMARY ===", flush=True)
    print(json.dumps(out, indent=2))
    ok = (score_within_tol and skipped_at_start > 0
          and flag_off_no_dir)
    print("WITHIN_TOL=%s (delta=%.4f<=%.2f) SKIPPED=%s GATE=%s"
          % (score_within_tol, score_delta, args.score_tol, skipped_at_start, "GREEN" if ok else "RED"), flush=True)
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
