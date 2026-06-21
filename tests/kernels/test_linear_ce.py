"""Fast GEMM-based fused linear CE (Liger-style): correctness vs numpy + perf vs the
slow streaming-fused path (5 TFLOPS fwd / 1 TFLOPS bwd)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.loss import linear_ce, GTM, GTN, GTK, CTM, TV

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def ref(hidden, w_head, labels, adv):
    M, V = hidden.shape[0], w_head.shape[1]
    logits = hidden.astype(np.float64) @ w_head.astype(np.float64)
    mx = logits.max(-1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(-1))
    logprob = logits[np.arange(M), labels] - lse
    p = np.exp(logits - mx); p /= p.sum(-1, keepdims=True)
    onehot = np.zeros((M, V)); onehot[np.arange(M), labels] = 1.0
    glogit = (adv[:, None] / M) * (p - onehot)
    return logprob, glogit @ w_head.T, hidden.T @ glogit


def test_correctness():
    print("--- correctness (vs numpy float64) ---")
    rng = np.random.default_rng(0); ok = True
    for (M, H, V) in [(128, 256, 512), (512, 512, 2048)]:
        hidden = (rng.standard_normal((M, H)) * 0.3).astype(np.float32)
        w_head = (rng.standard_normal((H, V)) * 0.3).astype(np.float32)
        labels = rng.integers(0, V, M)
        adv    = rng.standard_normal(M).astype(np.float32)
        lp, dh, dw = linear_ce(hidden, w_head, labels, si, advantage=adv)
        lp_r, dh_r, dw_r = ref(hidden, w_head, labels, adv)
        def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
        rl, rh, rw = rel(lp, lp_r), rel(dh, dh_r), rel(dw, dw_r)
        o = rl < 0.03 and rh < 0.03 and rw < 0.03; ok &= o
        print(f"  M={M} H={H} V={V}: logprob={rl*100:.2f}% dhidden={rh*100:.2f}% "
              f"dW={rw*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def bench(M, H, V):
    """Kernel-only timing (CUDA events, pre-uploaded data) — excludes host overhead."""
    from ancora.kernels.loss import (_gemm, _ce_stats, _ce_grad, _GpuArray,
                                      f32_to_bf16_bits as f32bf)
    rng = np.random.default_rng(1)
    hidden = (rng.standard_normal((M, H)) * 0.3).astype(np.float32)
    w_head = (rng.standard_normal((H, V)) * 0.3).astype(np.float32)
    labels = rng.integers(0, V, M).astype(np.int32); adv = rng.standard_normal(M).astype(np.float32)
    gh = _GpuArray(f32bf(hidden)); gw = _GpuArray(f32bf(w_head))
    glab = _GpuArray(labels.reshape(M, 1)); ga = _GpuArray(adv.reshape(M, 1))
    gLg = _GpuArray.zeros((M, V), np.float32); gp = _GpuArray.zeros((M, 1), np.float32)
    gls = _GpuArray.zeros((M, 1), np.float32); gG = _GpuArray.zeros((M, V), np.uint16)
    gwT = _GpuArray(f32bf(np.ascontiguousarray(w_head.T))); ghT = _GpuArray(f32bf(np.ascontiguousarray(hidden.T)))
    gdh = _GpuArray.zeros((M, H), np.float32); gdw = _GpuArray.zeros((H, V), np.float32)

    def run():
        ct.launch(si, (M//GTM, V//GTN, 1), _gemm, (gh, gw, gLg, H//GTK, GTM, GTN, GTK))
        ct.launch(si, (M//CTM, 1, 1), _ce_stats, (gLg, glab, gp, gls, V//TV))
        ct.launch(si, (M//CTM, 1, 1), _ce_grad, (gLg, gls, glab, ga, gG, V//TV, 1.0/M))
        ct.launch(si, (M//GTM, H//GTN, 1), _gemm, (gG, gwT, gdh, V//GTK, GTM, GTN, GTK))
        ct.launch(si, (H//GTM, V//GTN, 1), _gemm, (ghT, gG, gdw, M//GTK, GTM, GTN, GTK))

    for _ in range(5): run()
    stream_obj.sync()
    _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, stream_obj.__cuda_stream__()[1])
    for _ in range(20): run()
    cudart.cudaEventRecord(t1, stream_obj.__cuda_stream__()[1]); cudart.cudaEventSynchronize(t1)
    _, ms = cudart.cudaEventElapsedTime(t0, t1); ms /= 20
    gemm = 2.0 * M * V * H
    for g in (gh, gw, glab, ga, gLg, gp, gls, gG, gwT, ghT, gdh, gdw): g.free()
    print(f"  M={M} H={H} V={V}: {ms:.2f} ms kernels (fwd+bwd)  "
          f"~{3*gemm/(ms/1e3)/1e12:.0f} TFLOPS  (slow-fused was ~1-6)")


if __name__ == "__main__":
    print(f"fast linear-CE — GEMM {GTM}/{GTN}/{GTK}, CTM={CTM} TV={TV}")
    print("=" * 60)
    ok = test_correctness()
    print("--- perf (fwd+bwd) ---")
    bench(4096, 1024, 8192)
    bench(2048, 1024, 16384)
    print("=" * 60)
    print(f"  correctness {'PASS' if ok else 'FAIL'}")
