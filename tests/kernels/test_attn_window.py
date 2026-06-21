"""Sliding-window (LOCAL) attention — fwd + bwd vs fp64 windowed reference, plus the
RL-critical seq-len invariance (a query at position t sees a FIXED window → identical
output regardless of total S). Window = causal AND i-j < window.

Keep: re-run after any attention-kernel change."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import flash_attn_forward, flash_attn_backward, D, BQ
from ancora.kernels.attention import _f32_to_bf16_bits as f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()                 # keep a ref — else the stream is GC'd (invalid handle)
si = int(stream_obj.__cuda_stream__()[1])


def _mask(S, window):
    i = np.arange(S)[:, None]; j = np.arange(S)[None, :]
    return (j <= i) & (i - j < window)          # causal AND within sliding window

def ref_fwd_window(Q, K, V, window):
    B, Hq, S, d = Q.shape; Hkv = K.shape[1]; G = Hq // Hkv; scale = 1.0 / math.sqrt(d)
    Qd, Kd, Vd = (a.astype(np.float64) for a in (Q, K, V)); keep = _mask(S, window)
    O = np.zeros((B, Hq, S, d))
    for b in range(B):
        for h in range(Hq):
            kv = h // G
            Sc = np.where(keep, (Qd[b, h] @ Kd[b, kv].T) * scale, -1e38)
            P = np.exp(Sc - Sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
            O[b, h] = P @ Vd[b, kv]
    return O

def ref_bwd_window(Q, K, V, dO, window):
    B, Hq, S, d = Q.shape; Hkv = K.shape[1]; G = Hq // Hkv; scale = 1.0 / math.sqrt(d)
    Qd, Kd, Vd, dOd = (a.astype(np.float64) for a in (Q, K, V, dO)); keep = _mask(S, window)
    dQ = np.zeros_like(Qd); dK = np.zeros_like(Kd); dV = np.zeros_like(Vd)
    for b in range(B):
        for h in range(Hq):
            kv = h // G
            Sc = np.where(keep, (Qd[b, h] @ Kd[b, kv].T) * scale, -1e38)
            P = np.exp(Sc - Sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
            do = dOd[b, h]
            dV[b, kv] += P.T @ do
            dP = do @ Vd[b, kv].T
            Dlt = (do * (P @ Vd[b, kv])).sum(-1, keepdims=True)
            dS = P * (dP - Dlt)
            dQ[b, h]  = (dS @ Kd[b, kv]) * scale
            dK[b, kv] += (dS.T @ Qd[b, h]) * scale
    return dQ, dK, dV

def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def test_forward():
    print("--- windowed forward vs fp64 ---")
    rng = np.random.default_rng(0); ok = True
    for (B, Hq, Hkv, S, W) in [(1, 1, 1, 512, 128), (1, 16, 8, 512, 128),
                               (2, 16, 8, 512, 256), (1, 16, 8, 256, 64)]:
        Q = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5
        K = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        V = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        O = flash_attn_forward(Q, K, V, si, window=W)
        e = rel(O, ref_fwd_window(Q, K, V, W)); o = e < 0.02; ok &= o
        print(f"  B={B} Hq={Hq} S={S} W={W}: rel={e*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok

def test_backward():
    print("--- windowed backward vs fp64 ---")
    rng = np.random.default_rng(1); ok = True
    for (B, Hq, Hkv, S, W) in [(1, 1, 1, 512, 128), (1, 16, 8, 512, 128), (2, 16, 8, 256, 128)]:
        Q  = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5
        K  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        V  = rng.standard_normal((B, Hkv, S, D)).astype(np.float32) * 0.5
        dO = rng.standard_normal((B, Hq,  S, D)).astype(np.float32) * 0.5
        O, L = flash_attn_forward(Q, K, V, si, return_lse=True, window=W)
        dQ, dK, dV = flash_attn_backward(Q, K, V, O, dO, L, si, window=W)
        dQr, dKr, dVr = ref_bwd_window(Q, K, V, dO, W)
        rq, rk, rv = rel(dQ, dQr), rel(dK, dKr), rel(dV, dVr)
        o = rq < 0.03 and rk < 0.03 and rv < 0.03; ok &= o
        print(f"  B={B} Hq={Hq} S={S} W={W}: dQ={rq*100:.2f}% dK={rk*100:.2f}% dV={rv*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok

def test_seqlen_invariance():
    """A query at position t sees a fixed window [t-W+1, t] → its output must be bitwise
    identical for S=256 vs S=512 (the rollout==training property, with sliding window)."""
    print("--- windowed seq-len invariance (token t bitwise S=256 vs 512) ---")
    rng = np.random.default_rng(2); Hq, Hkv, W = 16, 8, 128
    Q = rng.standard_normal((1, Hq,  512, D)).astype(np.float32) * 0.5
    K = rng.standard_normal((1, Hkv, 512, D)).astype(np.float32) * 0.5
    V = rng.standard_normal((1, Hkv, 512, D)).astype(np.float32) * 0.5
    O512 = flash_attn_forward(Q, K, V, si, window=W)
    O256 = flash_attn_forward(Q[:, :, :256], K[:, :, :256], V[:, :, :256], si, window=W)
    # positions 0..255 must match bitwise (compared at BF16 store precision)
    same = np.array_equal(f32bf(O512[:, :, :256]), f32bf(O256))
    print(f"  tokens 0..255 bitwise identical: {same}  {'OK' if same else 'FAIL'}")
    return same

def test_full_causal_regression():
    """window >= S must equal full causal (the existing kernel)."""
    print("--- window>=S == full causal ---")
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((1, 16, 256, D)).astype(np.float32) * 0.5
    K = rng.standard_normal((1, 8,  256, D)).astype(np.float32) * 0.5
    V = rng.standard_normal((1, 8,  256, D)).astype(np.float32) * 0.5
    Ofull = flash_attn_forward(Q, K, V, si, window=0)
    Owin  = flash_attn_forward(Q, K, V, si, window=256)     # window spans whole seq
    same = np.array_equal(f32bf(Ofull), f32bf(Owin))
    print(f"  window=256 (==S) matches full causal bitwise: {same}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    print(f"sliding-window attention — BQ={BQ} D={D}")
    print("=" * 64)
    r = [test_forward(), test_backward(), test_seqlen_invariance(), test_full_causal_regression()]
    print("=" * 64)
    print("  ALL PASS" if all(r) else "  SOME FAILED: " + str(r))
