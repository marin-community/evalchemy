#!/usr/bin/env python
"""Stage-6 GPU smoke: chat_benchmark resume integration via the REAL MATH500 path.

Drives ``MATH500Benchmark.compute`` (greedy / do_sample=False) through the resume
wrap on a small problem subset + a small model.

PARITY CONTRACT (settled, see resume_manager_plan.md invariant #2 + the Stage-3c
"within sampling noise" decision, and the batch-composition probe 46943624):

  vLLM 0.11.2 greedy generation is NOT invariant to BATCH COMPOSITION even with
  VLLM_BATCH_INVARIANT=1 — a prompt generated in a batch-of-K differs from the SAME
  prompt in a batch-of-M (probe: A-alone != B-in-batch, 529 vs 531 chars; prefix
  caching ON/OFF identical). Resume necessarily regenerates a DIFFERENT-shaped batch
  than an uninterrupted batch-of-N run, so a naive resumed-vs-full-N comparison is
  byte-different for a reason that has NOTHING to do with the resume mechanism.

  We therefore split the gate into two honest checks:

  (1) MECHANISM PROOF (load-bearing, byte-exact): build a BATCH-SHAPE-MATCHED
      baseline = generate problems [0:kill] in a batch-of-kill PLUS problems
      [kill:N] in a batch-of-(N-kill), concatenated. This is EXACTLY the batch
      composition the resume path produces (partial-phase batch-of-kill for the
      restored units + resume-phase batch-of-(N-kill) for the regenerated units).
      The resumed per-problem outputs MUST be byte-identical to this matched
      baseline. If they are, the resume skip/restore/regenerate/merge machinery is
      provably exact; any score delta vs full-N is purely the vLLM batch artifact.

  (2) FULL-N REFERENCE (reported, within named tolerance): the uninterrupted
      batch-of-N baseline. Its score may differ from the resumed score by the vLLM
      batch-composition artifact; we assert |Δaccuracy| <= --acc-tol and REPORT the
      byte diff (named, not silently loosened).

  GREEN = matched-baseline byte-identical (mechanism exact) AND skipped == kill > 0
          AND |Δacc vs full-N| <= tol.
"""

import argparse
import json
import os
import sys

import lm_eval.api.registry

import eval.lm_eval_compat  # noqa: F401,E402
import lm_eval.models.openai_completions  # noqa: F401,E402
import lm_eval.models.vllm_causallms  # noqa: F401,E402
import lm_eval.models.huggingface  # noqa: F401,E402

from eval.resume import ResumeManager, RunFingerprint, read_manifest


def build_lm(model_args: str):
    return lm_eval.api.registry.get_model("vllm").create_from_arg_string(model_args, {"device": None})


def write_subset(rows, dst):
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return dst


def make_manager(run_dir, subset, mode="auto"):
    fp = RunFingerprint.from_run_inputs(
        model_repo="Qwen/Qwen3-0.6B",
        task_name="MATH500",
        task_data_path=subset,
        grader_version="hendrycks_math_v1",
        decoding={"do_sample": False, "temperature": 0.7, "max_new_tokens": 2048},
        seed_set=[0, 1234, 1234, 1234],
    )
    return ResumeManager(run_dir=run_dir, fingerprint=fp, mode=mode, world_size=1, rank=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--kill-after", type=int, default=15)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--acc-tol", type=float, default=0.20,
                    help="tolerance on |resumed_acc - full_N_baseline_acc| (vLLM batch-composition artifact)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(repo_root, "eval/chat_benchmarks/MATH500/data/math500.jsonl")
    with open(src) as f:
        all_rows = [json.loads(x) for x in f][: args.limit]

    full_subset = write_subset(all_rows, os.path.join(args.out, "math500_full.jsonl"))
    half_subset = write_subset(all_rows[: args.kill_after], os.path.join(args.out, "math500_half.jsonl"))
    rest_subset = write_subset(all_rows[args.kill_after:], os.path.join(args.out, "math500_rest.jsonl"))

    from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark

    model_args = (
        f"pretrained={args.model_repo},tensor_parallel_size={args.tp},"
        f"dtype=bfloat16,max_model_len={args.max_model_len},gpu_memory_utilization=0.9,seed=42"
    )
    print(f">>> building vLLM LM: {model_args}", flush=True)
    lm = build_lm(model_args)

    def run_once(bench):
        gen = bench.generate_responses(lm)
        res = bench.evaluate_responses(gen)
        if lm.rank != 0:
            return None
        outs = [(ex["model_output"], ex["model_answer"]) for ex in res["examples"]]
        return {"accuracy": res["accuracy"], "num_solved": res["num_solved"],
                "num_total": res["num_total"], "outputs": outs}

    # ---- full-N baseline: no manager (today's path), one batch-of-N ----
    print(">>> BASELINE full-N (no manager, batch-of-N)", flush=True)
    base = run_once(MATH500Benchmark(data_file=full_subset, max_tokens=args.max_tokens))

    # ---- batch-shape-MATCHED baseline: [0:kill] in batch-of-kill, [kill:N] in batch-of-(N-kill) ----
    # This reproduces the EXACT batch composition the resume path generates, so the
    # resumed outputs must be byte-identical to it (the mechanism proof).
    print(">>> MATCHED baseline (batch-of-kill + batch-of-rest, no manager)", flush=True)
    m_half = run_once(MATH500Benchmark(data_file=half_subset, max_tokens=args.max_tokens))
    m_rest = run_once(MATH500Benchmark(data_file=rest_subset, max_tokens=args.max_tokens))
    matched_outputs = m_half["outputs"] + m_rest["outputs"]

    # ---- partial: manager (auto, fresh) over the first kill_after problems ----
    run_dir = os.path.join(args.out, "resume_run")
    if os.path.exists(os.path.join(run_dir, "resume")):
        import shutil
        shutil.rmtree(os.path.join(run_dir, "resume"))
    print(f">>> PARTIAL (first {args.kill_after} problems) run_dir={run_dir}", flush=True)
    bench_p = MATH500Benchmark(data_file=half_subset, max_tokens=args.max_tokens)
    mgr_p = make_manager(run_dir, full_subset, mode="auto")
    assert mgr_p.decide() == "fresh", "partial run should start fresh"
    bench_p.attach_resume_manager(mgr_p)
    run_once(bench_p)  # records kill_after units to the manifest, then the process "dies"
    n_done_partial = len(read_manifest(os.path.join(run_dir, "resume", "manifest.jsonl")))
    print(f">>> partial manifest has {n_done_partial} done units", flush=True)

    # ---- resume: manager (auto) on the same run dir over ALL N ----
    print(">>> RESUME (all problems, manager detects prior state)", flush=True)
    bench_r = MATH500Benchmark(data_file=full_subset, max_tokens=args.max_tokens)
    mgr_r = make_manager(run_dir, full_subset, mode="auto")
    assert mgr_r.decide() == "resume", "resume run should detect prior state"
    bench_r.attach_resume_manager(mgr_r)

    gen_calls = {"n": 0}
    orig_generate = lm.generate_until

    def counting_generate(instances):
        gen_calls["n"] += len(instances)
        return orig_generate(instances)

    lm.generate_until = counting_generate
    try:
        resumed = run_once(bench_r)
    finally:
        lm.generate_until = orig_generate

    skipped = args.limit - gen_calls["n"]
    print(f">>> resume regenerated {gen_calls['n']} problems, skipped {skipped}", flush=True)

    # ---- compare ----
    # (1) MECHANISM PROOF: resumed == batch-shape-matched baseline, byte-exact.
    matched_byte_identical = resumed["outputs"] == matched_outputs
    # (2) FULL-N REFERENCE: reported; score within named tolerance.
    full_byte_identical = resumed["outputs"] == base["outputs"]
    acc_delta_vs_full = abs(resumed["accuracy"] - base["accuracy"])
    acc_within_tol = acc_delta_vs_full <= args.acc_tol

    # locate first mechanism-proof divergence (should be none)
    first_mismatch = None
    if not matched_byte_identical:
        for i, (r, m) in enumerate(zip(resumed["outputs"], matched_outputs)):
            if r != m:
                first_mismatch = {"index": i, "in_skipped_half": i < args.kill_after}
                break

    out = {
        "model": args.model_repo, "benchmark": "MATH500", "limit": args.limit,
        "kill_after": args.kill_after,
        "baseline_full_accuracy": base["accuracy"], "baseline_full_num_solved": base["num_solved"],
        "resumed_accuracy": resumed["accuracy"], "resumed_num_solved": resumed["num_solved"],
        "partial_done_units": n_done_partial,
        "resume_regenerated": gen_calls["n"], "resume_skipped": skipped,
        "matched_baseline_byte_identical": bool(matched_byte_identical),   # (1) load-bearing
        "matched_first_mismatch": first_mismatch,
        "full_N_byte_identical": bool(full_byte_identical),                # reported
        "acc_delta_vs_full_N": acc_delta_vs_full,
        "acc_tol": args.acc_tol,
        "acc_within_tol_vs_full_N": bool(acc_within_tol),
    }
    with open(os.path.join(args.out, "smoke_resume_chat.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=== STAGE 3a/6 SMOKE SUMMARY ===", flush=True)
    print(json.dumps(out, indent=2))

    # GREEN gate:
    #   - mechanism exact (resumed byte-identical to the batch-shape-matched baseline)
    #   - skipped exactly kill_after, and > 0 (resume actually skipped completed units)
    #   - score within the named tolerance of the full-N reference
    ok = (matched_byte_identical
          and skipped == args.kill_after and skipped > 0
          and acc_within_tol)
    print(f"PASS={ok}  (matched_byte_identical={matched_byte_identical}, "
          f"skipped={skipped}, acc_delta_vs_full={acc_delta_vs_full:.4f}<= {args.acc_tol})", flush=True)
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
