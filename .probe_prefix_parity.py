#!/usr/bin/env python
"""Isolate WHY chat/gsm8k resume parity fails while pass@k passes.

Hypothesis: with VLLM_BATCH_INVARIANT=1, greedy per-sequence output should be
byte-identical regardless of (a) batch size/composition and (b) the order/history
of prior generate() calls -- UNLESS prefix caching reuses KV blocks computed under
a different batch context.

This builds ONE LLM and, for a fixed set of greedy prompts, checks byte-identity of
a target prompt's output across three regimes, with prefix caching ON then OFF:

  regime A: target generated ALONE (fresh engine state)
  regime B: target generated in a BATCH with other prompts (baseline-like)
  regime C: target generated AFTER a prior batch of *other* prompts ran first
            (resume-like: cache/RNG history polluted by the partial phase)

If A==B==C with caching ON  -> prefix caching is innocent; look elsewhere.
If they differ with caching ON but all match with caching OFF -> prefix caching
is the resume-parity breaker; fix = enable_prefix_caching=False for parity runs.
"""
import argparse, os
from vllm import LLM, SamplingParams

PROMPTS = [
    "Question: What is 17 times 24? Answer step by step.\nAnswer:",
    "Question: A train travels 60 miles in 1.5 hours. What is its speed?\nAnswer:",
    "Question: If a rectangle has length 8 and width 3, what is its area?\nAnswer:",
    "Question: Solve for x: 3x + 7 = 22.\nAnswer:",
    "Question: What is the sum of the first 10 positive integers?\nAnswer:",
    "Question: A shirt costs $40 and is discounted 25%. What is the sale price?\nAnswer:",
]
TARGET = PROMPTS[2]  # the held-out probe prompt
OTHERS = [p for p in PROMPTS if p != TARGET]


def gen(llm, prompts):
    sp = SamplingParams(temperature=0.0, max_tokens=200, seed=None)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return {o.prompt: o.outputs[0].text for o in outs}


def run_regimes(prefix_caching: bool, model: str, mml: int):
    print(f"\n========== enable_prefix_caching={prefix_caching} ==========", flush=True)
    llm = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=mml, gpu_memory_utilization=0.55, seed=42,
              enable_prefix_caching=prefix_caching, enforce_eager=False)

    # regime A: target alone, fresh
    a = gen(llm, [TARGET])[TARGET]
    # regime B: target in a batch with others (one call)
    b = gen(llm, OTHERS[:3] + [TARGET] + OTHERS[3:])[TARGET]
    # regime C: run a *prior* batch of others, THEN target alone (resume-like history)
    gen(llm, OTHERS)            # pollute cache + RNG history
    c = gen(llm, [TARGET])[TARGET]

    print(f"len(A)={len(a)} len(B)={len(b)} len(C)={len(c)}")
    print(f"A==B : {a==b}")
    print(f"A==C : {a==c}")
    print(f"B==C : {b==c}")
    if a != c:
        # show first divergence
        for i,(x,y) in enumerate(zip(a,c)):
            if x!=y:
                print(f"  A vs C first diff @char {i}: {a[max(0,i-20):i+20]!r}  vs  {c[max(0,i-20):i+20]!r}")
                break
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return a==b and a==c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    on_ok = run_regimes(True, args.model, args.max_model_len)
    off_ok = run_regimes(False, args.model, args.max_model_len)
    print("\n================ VERDICT ================")
    print(f"prefix_caching=ON  all-regimes-byte-identical: {on_ok}")
    print(f"prefix_caching=OFF all-regimes-byte-identical: {off_ok}")
    if not on_ok and off_ok:
        print(">>> CONFIRMED: prefix caching breaks greedy batch-invariant parity; fix=disable it for parity runs.")
    elif on_ok:
        print(">>> Prefix caching innocent; resume parity gap is elsewhere.")
    else:
        print(">>> Neither regime is byte-identical; deeper nondeterminism (investigate further).")


if __name__ == "__main__":
    main()
