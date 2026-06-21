"""
Batch-invariance / train-inference consistency tests for attention.

The RL-critical property: a token at position t must get the SAME output
(→ same logprob) regardless of:
  (1) how many other sequences are in the batch  (batch-size invariance)
  (2) the total sequence length                  (sequence-length invariance)

Reference: Thinking Machines "Defeating Nondeterminism in LLM Inference" (2025).
Root cause of mismatch = reduction order changing with batch size.
Our attention loops kv=0,1,...,NKVB-1 sequentially in one block (no split-KV),
so the reduction order is fixed → should be BITWISE identical.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import flash_attn_forward, D, BQ

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()   # keep a reference, else the stream is freed
stream_int = int(stream_obj.__cuda_stream__()[1])


def test_determinism():
    """Same input twice → bitwise identical (no race)."""
    rng = np.random.default_rng(0)
    B, Hq, Hkv, S = 2, 16, 8, 512
    Q = rng.standard_normal((B, Hq,  S, D)).astype(np.float32)
    K = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    V = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    O1 = flash_attn_forward(Q, K, V, stream_int)
    O2 = flash_attn_forward(Q, K, V, stream_int)
    ok = np.array_equal(O1, O2)
    print(f"  determinism (run twice):        bitwise {'IDENTICAL' if ok else 'DIFFERENT'}  {'OK' if ok else 'FAIL'}")
    return ok


def test_batch_size_invariance():
    """batch 0's output must be identical alone (B=1) vs in a batch (B=4)."""
    rng = np.random.default_rng(1)
    B, Hq, Hkv, S = 4, 16, 8, 512
    Q = rng.standard_normal((B, Hq,  S, D)).astype(np.float32)
    K = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    V = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)

    O_full   = flash_attn_forward(Q, K, V, stream_int)            # (4, Hq, S, D)
    O_single = flash_attn_forward(Q[:1], K[:1], V[:1], stream_int) # (1, Hq, S, D)

    ok = np.array_equal(O_full[0], O_single[0])
    diff = np.abs(O_full[0] - O_single[0]).max()
    print(f"  batch-size invariance (B=1 vs B=4): bitwise {'IDENTICAL' if ok else f'DIFFERENT maxdiff={diff:.2e}'}  {'OK' if ok else 'FAIL'}")
    return ok


def test_seq_len_invariance():
    """
    RL-critical: token at position t must get identical output whether the
    sequence is length 256 or 512 (causal → future tokens don't affect t).
    Simulates rollout (incremental) vs training (full-sequence) consistency.
    """
    rng = np.random.default_rng(2)
    B, Hq, Hkv = 1, 16, 8
    S_long = 512
    S_short = 256
    # Build a long sequence; the short one is its prefix
    Q = rng.standard_normal((B, Hq,  S_long, D)).astype(np.float32)
    K = rng.standard_normal((B, Hkv, S_long, D)).astype(np.float32)
    V = rng.standard_normal((B, Hkv, S_long, D)).astype(np.float32)

    O_long  = flash_attn_forward(Q, K, V, stream_int)                  # (1,Hq,512,D)
    O_short = flash_attn_forward(Q[:, :, :S_short], K[:, :, :S_short],
                                 V[:, :, :S_short], stream_int)        # (1,Hq,256,D)

    # positions 0..255 should match bitwise (causal: don't see 256..511)
    ok = np.array_equal(O_long[:, :, :S_short], O_short)
    diff = np.abs(O_long[:, :, :S_short] - O_short).max()
    print(f"  seq-len invariance (256 vs 512 prefix): bitwise {'IDENTICAL' if ok else f'DIFFERENT maxdiff={diff:.2e}'}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"Batch-invariance tests — attention BQ={BQ} D={D}")
    print("=" * 60)
    results = [
        test_determinism(),
        test_batch_size_invariance(),
        test_seq_len_invariance(),
    ]
    print("=" * 60)
    print(f"  {sum(results)}/{len(results)} passed"
          + ("  → kernel is batch-invariant ✓" if all(results) else "  → INVARIANCE LEAK"))
