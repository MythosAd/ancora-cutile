"""
Correctness + perf test for ancora/kernels/attention.py
Compares cuda-tile Flash Attention against numpy reference (causal GQA).
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

import ancora.env
from ancora.kernels.attention import flash_attn_forward, BQ, BKV, D, _attn_fwd, _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
stream_int = int(stream_obj.__cuda_stream__()[1])


def ref_attention(Q, K, V, causal=True):
    """Numpy reference: causal GQA attention. float64 for accuracy."""
    B, Hq, Sq, d = Q.shape
    _, Hkv, Skv, _ = K.shape
    G = Hq // Hkv
    scale = 1.0 / math.sqrt(d)
    O = np.zeros((B, Hq, Sq, d), np.float64)
    Qd, Kd, Vd = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    for b in range(B):
        for h in range(Hq):
            kv = h // G
            s = (Qd[b, h] @ Kd[b, kv].T) * scale          # (Sq, Skv)
            if causal:
                qp = np.arange(Sq)[:, None]
                kp = np.arange(Skv)[None, :]
                s = np.where(qp >= kp, s, -1e38)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(-1, keepdims=True)
            O[b, h] = p @ Vd[b, kv]
    return O


def test_correctness():
    print("--- Correctness (vs numpy float64 reference) ---")
    rng = np.random.default_rng(0)
    cases = [
        # (B, Hq, Hkv, Sq, Skv) — Sq,Skv multiples of BQ,BKV
        (1, 1, 1, 128, 128),   # single block, MHA
        (1, 2, 1, 256, 256),   # 2 blocks, GQA G=2
        (1, 16, 8, 512, 512),  # Qwen3-0.6B head config
        (2, 16, 8, 256, 256),  # batch=2
    ]
    all_ok = True
    for (B, Hq, Hkv, Sq, Skv) in cases:
        Q = rng.standard_normal((B, Hq,  Sq,  D)).astype(np.float32)
        K = rng.standard_normal((B, Hkv, Skv, D)).astype(np.float32)
        V = rng.standard_normal((B, Hkv, Skv, D)).astype(np.float32)

        O_ct  = flash_attn_forward(Q, K, V, stream_int, causal=True)
        O_ref = ref_attention(Q, K, V, causal=True)

        err = np.abs(O_ct.astype(np.float64) - O_ref).max()
        rel = err / (np.abs(O_ref).max() + 1e-9)
        ok  = rel < 0.02   # 2% — BF16 matmul tolerance
        all_ok &= ok
        print(f"  B={B} Hq={Hq} Hkv={Hkv} S={Sq:<4}  "
              f"max_abs_err={err:.4f}  rel={rel*100:.2f}%  {'OK' if ok else 'FAIL'}")
    return all_ok


def bench(B, H, Hkv, S, iters=50, warmup=10):
    """Benchmark TFLOPS. Causal attention FLOPs = 2 * 2 * B*H*S*S*D / 2 (causal halves)."""
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((B, H,   S, D)).astype(np.float32)
    K = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    V = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)

    # Pre-build GPU arrays once (exclude H2D from timing)
    from ancora.kernels.attention import _prep_qkv
    NQB, NKVB = S // BQ, S // BKV
    scale = float(1.0 / math.sqrt(D))
    gQ = _GpuArray(_prep_qkv(Q, NQB, BQ))
    gK = _GpuArray(_prep_qkv(K, NKVB, BKV))
    gV = _GpuArray(_prep_qkv(V, NKVB, BKV))
    gO = _GpuArray.zeros((B * H * NQB * BQ, D), np.float32)
    gL = _GpuArray.zeros((B * H * NQB * BQ, 1), np.float32)   # logsumexp output

    def launch():
        ct.launch(stream_int, (NQB, B * H, 1), _attn_fwd,
                  (gQ, gK, gV, gO, gL, NQB, NKVB, H, Hkv, scale))

    for _ in range(warmup): launch()
    stream_obj.sync()

    err, t0 = cudart.cudaEventCreate(); err, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, stream_obj.__cuda_stream__()[1])
    for _ in range(iters): launch()
    cudart.cudaEventRecord(t1, stream_obj.__cuda_stream__()[1])
    cudart.cudaEventSynchronize(t1)
    err, ms = cudart.cudaEventElapsedTime(t0, t1)

    # Causal FLOPs: 2 matmuls (QK^T, PV), each 2*S*S*D, times B*H, halved for causal
    flops = 2.0 * 2.0 * B * H * S * S * D * 0.5
    tflops = flops * iters / (ms / 1000) / 1e12
    for g in (gQ, gK, gV, gO, gL): g.free()
    print(f"  B={B} H={H} S={S:<5}  {ms/iters:.3f} ms/iter  {tflops:.1f} TFLOPS")
    return tflops


if __name__ == "__main__":
    print(f"ancora attention — BQ={BQ} BKV={BKV} D={D}  sm_120a")
    print("=" * 60)
    ok = test_correctness()
    print()
    print("--- Performance (causal, BF16) ---")
    for (B, H, Hkv, S) in [(1, 16, 8, 512), (1, 16, 8, 1024),
                            (4, 16, 8, 512), (1, 16, 8, 2048)]:
        bench(B, H, Hkv, S)
    print("=" * 60)
    print(f"  correctness: {'PASS' if ok else 'FAIL'}")
