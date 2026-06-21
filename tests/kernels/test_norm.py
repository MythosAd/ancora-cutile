"""RMSNorm forward/backward correctness (vs numpy fp64 on BF16-rounded inputs) +
batch invariance (per-row output is bitwise identical regardless of batch size).

Keep this around: re-run after every cuda-tile / toolkit upgrade (rule from
keep-test-benchmark-code)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.norm import (rmsnorm_forward, rmsnorm_backward,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32, TM)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def _bf(x):  # round f32 → bf16 value (through the bit pattern), as the kernel sees it
    return bf32(f32bf(x))


def ref_fwd(x, w, eps):
    xb, wb = _bf(x).astype(np.float64), _bf(w).astype(np.float64)
    rstd = 1.0 / np.sqrt((xb * xb).mean(-1, keepdims=True) + eps)
    return xb * rstd * wb, rstd


def ref_bwd(x, w, dy, rstd, H):
    xb, wb, dyb = _bf(x).astype(np.float64), _bf(w).astype(np.float64), _bf(dy).astype(np.float64)
    dyw = dyb * wb
    c = (dyw * xb).sum(-1, keepdims=True)
    dx = rstd * dyw - rstd**3 * xb * c / H
    dw = (dyb * xb * rstd).sum(0)
    return dx, dw


def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def test_correctness():
    print("--- correctness (vs numpy fp64, BF16-rounded inputs) ---")
    rng = np.random.default_rng(0); ok = True; eps = 1e-6
    for (M, H) in [(128, 128), (512, 1024), (4096, 1024)]:
        x = (rng.standard_normal((M, H)) * 0.8).astype(np.float32)
        w = (1.0 + rng.standard_normal(H) * 0.2).astype(np.float32)
        dy = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)

        y, rstd = rmsnorm_forward(x, w, si, eps)
        y_r, rstd_r = ref_fwd(x, w, eps)
        dx, dw = rmsnorm_backward(x, w, dy, rstd, si)
        dx_r, dw_r = ref_bwd(x, w, dy, rstd, H)

        ry, rr = rel(_bf(y), y_r), rel(rstd, rstd_r)
        rdx, rdw = rel(_bf(dx), dx_r), rel(dw, dw_r)
        o = ry < 0.03 and rr < 0.01 and rdx < 0.04 and rdw < 0.03; ok &= o
        print(f"  M={M:5d} H={H:5d}: y={ry*100:.2f}% rstd={rr*100:.3f}% "
              f"dx={rdx*100:.2f}% dw={rdw*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_batch_invariance():
    """Row 0's data is identical for M=TM vs M=8*TM → output must be bitwise identical
    (RMSNorm forward reduces only within a row; no cross-token dependence)."""
    print("--- batch invariance (row-0 output bitwise identical) ---")
    rng = np.random.default_rng(1); H = 1024; eps = 1e-6
    base = (rng.standard_normal((TM, H)) * 0.8).astype(np.float32)
    w = (1.0 + rng.standard_normal(H) * 0.2).astype(np.float32)

    def run(M):
        x = np.zeros((M, H), np.float32); x[:TM] = base
        x[TM:] = rng.standard_normal((M - TM, H)).astype(np.float32)  # different "other" rows
        y, rstd = rmsnorm_forward(x, w, si, eps)
        return f32bf(y[:TM]), rstd[:TM]   # compare bits

    y1, r1 = run(TM)
    y8, r8 = run(8 * TM)
    same = np.array_equal(y1, y8) and np.array_equal(r1, r8)
    print(f"  M={TM} vs M={8*TM}: y bits identical={np.array_equal(y1,y8)} "
          f"rstd identical={np.array_equal(r1,r8)}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    print(f"RMSNorm — TM={TM}")
    print("=" * 60)
    ok = test_correctness()
    bi = test_batch_invariance()
    print("=" * 60)
    print(f"  correctness {'PASS' if ok else 'FAIL'} | batch-invariance {'PASS' if bi else 'FAIL'}")
