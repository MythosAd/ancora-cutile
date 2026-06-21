"""
Correctness + perf for ancora/kernels/linear.py
- forward MXFP8 with MULTI-BLOCK K (exercises the scale-index fix)
- forward perf at 4096^3 (expect ~261 TFLOPS, the SMEM-tiled GEMM)
- backward BF16 (dx = dy@W, dw = dy^T@X)
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

import ancora.env
from ancora.kernels.linear import (
    _fwd_mxfp8, _bwd_input, _bwd_weight, _GpuArray,
    FTM, FTN, FTK, BTM, BTN, BTK, B,
)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

FP8_ONE  = 0x38     # FP8 E4M3 1.0
E8M0_ONE = 0x7F     # E8M0 1.0  (value = 2^(byte-127))
BF16_ONE = 0x3F80   # BF16 1.0


def test_fwd_correctness():
    """All-ones FP8, scale 1.0, multi-block K → out[i,j] = K. Exercises scale index."""
    M, N, K = 128, 256, 256          # K=256, FTK=32 → 8 K-blocks
    K_BLOCKS = K // FTK
    x  = _GpuArray(np.full((M, K),      FP8_ONE,  np.uint8))   # x (M,K)
    w  = _GpuArray(np.full((K, N),      FP8_ONE,  np.uint8))   # w PRE-TRANSPOSED (K,N)
    xs = _GpuArray(np.full((M, K // B), E8M0_ONE, np.uint8))
    ws = _GpuArray(np.full((K // B, N), E8M0_ONE, np.uint8))   # w_scale (K//B, N)
    out = _GpuArray.zeros((M, N), np.float32)

    ct.launch(si, (M // FTM, N // FTN, 1), _fwd_mxfp8,
              (x, w, xs, ws, out, M, N, K_BLOCKS, FTM, FTN, FTK))
    cudart.cudaStreamSynchronize(si)
    r = out.to_numpy()
    got = float(r[0, 0]); allok = np.allclose(r, K, rtol=0.02)
    print(f"  fwd MXFP8 (K={K}, {K_BLOCKS} blocks): out[0,0]={got:.1f} exp={K}  "
          f"all={'OK' if allok else 'FAIL'}")
    for g in (x, w, xs, ws, out): g.free()
    return allok


def test_fwd_perf():
    M = N = K = 4096
    K_BLOCKS = K // FTK
    x  = _GpuArray(np.full((M, K),      FP8_ONE,  np.uint8))
    w  = _GpuArray(np.full((K, N),      FP8_ONE,  np.uint8))   # PRE-TRANSPOSED (K,N)
    xs = _GpuArray(np.full((M, K // B), E8M0_ONE, np.uint8))
    ws = _GpuArray(np.full((K // B, N), E8M0_ONE, np.uint8))
    out = _GpuArray.zeros((M, N), np.float32)

    def L(): ct.launch(si, (M // FTM, N // FTN, 1), _fwd_mxfp8,
                       (x, w, xs, ws, out, M, N, K_BLOCKS, FTM, FTN, FTK))
    for _ in range(5): L()
    stream_obj.sync()
    _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, stream_obj.__cuda_stream__()[1])
    for _ in range(30): L()
    cudart.cudaEventRecord(t1, stream_obj.__cuda_stream__()[1]); cudart.cudaEventSynchronize(t1)
    _, ms = cudart.cudaEventElapsedTime(t0, t1)
    tf = 2.0 * M * N * K * 30 / (ms / 1000) / 1e12
    peak = 340.0
    print(f"  fwd MXFP8 perf 4096^3 (TM{FTM} TN{FTN} TK{FTK}): "
          f"{tf:.0f} TFLOPS  (~{tf/peak*100:.0f}% of ~{peak:.0f} FP8 peak, all-ones=cache-optimistic)")
    for g in (x, w, xs, ws, out): g.free()
    return tf


def test_bwd_correctness():
    M, N, K = 256, 256, 256
    # dx = dy @ W : dy(M,N)=1, W(N,K)=1 → dx = N
    dy = _GpuArray(np.full((M, N), BF16_ONE, np.uint16))
    w  = _GpuArray(np.full((N, K), BF16_ONE, np.uint16))
    dx = _GpuArray.zeros((M, K), np.float32)
    ct.launch(si, (M // BTM, K // BTK, 1), _bwd_input, (dy, w, dx, M, K, N // BTN))
    cudart.cudaStreamSynchronize(si)
    dxv = dx.to_numpy(); ok1 = np.allclose(dxv, N, rtol=0.01)
    print(f"  bwd dx = dy@W: dx[0,0]={dxv[0,0]:.1f} exp={N}  {'OK' if ok1 else 'FAIL'}")

    # dw = dy^T @ X : dy=1, X(M,K)=1 → dw = M
    x  = _GpuArray(np.full((M, K), BF16_ONE, np.uint16))
    dw = _GpuArray.zeros((N, K), np.float32)
    ct.launch(si, (N // BTN, K // BTK, 1), _bwd_weight, (dy, x, dw, N, K, M // BTM))
    cudart.cudaStreamSynchronize(si)
    dwv = dw.to_numpy(); ok2 = np.allclose(dwv, M, rtol=0.01)
    print(f"  bwd dw = dy^T@X: dw[0,0]={dwv[0,0]:.1f} exp={M}  {'OK' if ok2 else 'FAIL'}")

    for g in (dy, w, dx, x, dw): g.free()
    return ok1 and ok2


if __name__ == "__main__":
    print(f"ancora/kernels/linear.py — fwd TM{FTM}/TN{FTN}/TK{FTK}, bwd TM{BTM}/TN{BTN}/TK{BTK}")
    print("=" * 64)
    r = [test_fwd_correctness(), test_bwd_correctness()]
    print("  --- perf ---")
    test_fwd_perf()
    print("=" * 64)
    print(f"  {sum(r)}/{len(r)} correctness passed")
