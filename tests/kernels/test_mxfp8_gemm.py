"""MXFP8 forward GEMM: E4M3 encoder correctness + quantize→_fwd_mxfp8 accuracy vs f32
reference + TFLOPS. Gate before wiring MXFP8 into the model forward."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

import ancora.env
from ancora.kernels.linear import _fwd_mxfp8, FTM, FTN, FTK, _GpuArray
from ancora.kernels.quant import (f32_to_e4m3, quantize_rowblock, quantize_colblock,
                                   dequantize_rowblock, B)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def test_encoder():
    print("--- E4M3 encoder (known values) ---")
    cases = {1.0: 0x38, 2.0: 0x40, 0.5: 0x30, 448.0: 0x7E, -2.0: 0xC0, 0.0: 0x00, 4.0: 0x48}
    ok = True
    for v, expect in cases.items():
        got = int(f32_to_e4m3(np.array([v], np.float32))[0])
        o = got == expect; ok &= o
        print(f"  {v:>7} → 0x{got:02X} (want 0x{expect:02X})  {'OK' if o else 'FAIL'}")
    return ok


def bench_and_acc(M=512, K=1024, N=1024):
    print(f"--- MXFP8 GEMM accuracy + TFLOPS (M={M} K={K} N={N}) ---")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)        # (K,N) pre-transposed
    xf, xs = quantize_rowblock(x); wf, ws = quantize_colblock(w)

    gx = _GpuArray(xf); gw = _GpuArray(wf); gxs = _GpuArray(xs); gws = _GpuArray(ws)
    go = _GpuArray.zeros((M, N), np.float32)
    KB = K // FTK
    def run(): ct.launch(si, (M // FTM, N // FTN, 1), _fwd_mxfp8, (gx, gw, gxs, gws, go, M, N, KB, FTM, FTN, FTK))
    run(); cudart.cudaStreamSynchronize(si)
    y = go.to_numpy()

    ref = x.astype(np.float64) @ w.astype(np.float64)                 # true f32 matmul
    deq = dequantize_rowblock(xf, xs).astype(np.float64) @ \
          dequantize_rowblock(wf.T, ws.T).T.astype(np.float64)        # ideal-MXFP8 (kernel target)
    rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    e_ref, e_deq = rel(y, ref), rel(y, deq)
    print(f"  kernel vs f32 matmul: {e_ref*100:.2f}%   kernel vs dequant-matmul: {e_deq*100:.3f}%")

    for _ in range(5): run()
    cudart.cudaStreamSynchronize(si)
    _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, si)
    for _ in range(30): run()
    cudart.cudaEventRecord(t1, si); cudart.cudaEventSynchronize(t1)
    ms = cudart.cudaEventElapsedTime(t0, t1)[1] / 30
    tf = 2.0 * M * N * K / (ms / 1e3) / 1e12
    print(f"  {ms:.3f} ms  {tf:.0f} TFLOPS  (BF16 _gemm was ~54-76 at layer shapes)")
    for g in (gx, gw, gxs, gws, go): g.free()
    return e_ref < 0.05 and e_deq < 0.01


if __name__ == "__main__":
    print("MXFP8 forward GEMM"); print("=" * 60)
    ok = test_encoder()
    ok &= bench_and_acc(512, 1024, 1024)
    ok &= bench_and_acc(4096, 4096, 4096)   # the headline shape
    print("=" * 60)
    print(f"  {'PASS' if ok else 'FAIL'}")
