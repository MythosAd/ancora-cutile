"""
Correctness + batch-invariance for attention backward (dQ, dK, dV).
Two-kernel batch-invariant split (no atomics). vs numpy float64 reference.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import flash_attn_forward, flash_attn_backward, D, BQ

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def ref_bwd(Q, K, V, dO):
    """numpy float64 causal-GQA attention backward."""
    B, Hq, Sq, d = Q.shape
    _, Hkv, Skv, _ = K.shape
    G = Hq // Hkv; scale = 1.0 / math.sqrt(d)
    Qd, Kd, Vd, dOd = (a.astype(np.float64) for a in (Q, K, V, dO))
    dQ = np.zeros_like(Qd); dK = np.zeros_like(Kd); dV = np.zeros_like(Vd)
    mask = np.arange(Sq)[:, None] >= np.arange(Skv)[None, :]
    for b in range(B):
        for h in range(Hq):
            kv = h // G
            S = (Qd[b, h] @ Kd[b, kv].T) * scale
            S = np.where(mask, S, -1e38)
            P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
            do = dOd[b, h]
            dV[b, kv] += P.T @ do
            dP = do @ Vd[b, kv].T
            Dlt = (do * (P @ Vd[b, kv])).sum(-1, keepdims=True)
            dS = P * (dP - Dlt)
            dQ[b, h]  = (dS @ Kd[b, kv]) * scale
            dK[b, kv] += (dS.T @ Qd[b, h]) * scale
    return dQ, dK, dV


def test_correctness():
    print("--- backward correctness (vs numpy float64) ---")
    rng = np.random.default_rng(0)
    ok = True
    for (B, Hq, Hkv, S) in [(1, 1, 1, 128), (1, 2, 1, 128), (1, 16, 8, 256), (2, 16, 8, 128)]:
        Q  = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5
        K  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        V  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        dO = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5

        O, L = flash_attn_forward(Q, K, V, si, return_lse=True)
        dQ, dK, dV = flash_attn_backward(Q, K, V, O, dO, L, si)
        dQr, dKr, dVr = ref_bwd(Q, K, V, dO)

        def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
        rq, rk, rv = rel(dQ, dQr), rel(dK, dKr), rel(dV, dVr)
        o = rq < 0.03 and rk < 0.03 and rv < 0.03; ok &= o
        print(f"  B={B} Hq={Hq} Hkv={Hkv} S={S}: dQ={rq*100:.2f}% dK={rk*100:.2f}% "
              f"dV={rv*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_invariance():
    print("--- backward batch invariance (bitwise dK/dV) ---")
    rng = np.random.default_rng(1)
    B, Hq, Hkv, S = 4, 16, 8, 256
    Q  = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5
    K  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
    V  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
    dO = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5

    O, L = flash_attn_forward(Q, K, V, si, return_lse=True)
    dQ_f, dK_f, dV_f = flash_attn_backward(Q, K, V, O, dO, L, si)
    # batch 0 alone
    O1, L1 = flash_attn_forward(Q[:1], K[:1], V[:1], si, return_lse=True)
    dQ_1, dK_1, dV_1 = flash_attn_backward(Q[:1], K[:1], V[:1], O1, dO[:1], L1, si)

    ok = (np.array_equal(dQ_f[0], dQ_1[0]) and np.array_equal(dK_f[0], dK_1[0])
          and np.array_equal(dV_f[0], dV_1[0]))
    print(f"  batch 0 grads (alone vs in B=4): "
          f"{'bitwise IDENTICAL' if ok else 'DIFFER'}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"attention backward — BQ={BQ} D={D}  (two-kernel batch-invariant)")
    print("=" * 60)
    r = [test_correctness(), test_invariance()]
    print("=" * 60)
    print(f"  {sum(r)}/{len(r)} passed"
          + ("  → dQ/dK/dV correct + batch-invariant ✓" if all(r) else "  → FAIL"))
