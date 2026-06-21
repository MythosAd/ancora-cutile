"""On-GPU MXFP8 activation quant (_quant_mxfp8): dequant ≈ input, end-to-end GEMM vs f32,
and quant overhead vs the GEMM (must be small to keep MXFP8's ~2× GEMM win)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

import ancora.env
from ancora.kernels.linear import _fwd_mxfp8, FTM, FTN, FTK, _GpuArray
from ancora.kernels.quant import _quant_mxfp8, QTM, B, quantize_colblock, dequantize_rowblock

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
_bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
_bfval = lambda x: (_bf(x).astype(np.uint32) << 16).view(np.float32)


def time_ms(run, it=30):
    for _ in range(8): run()
    cudart.cudaStreamSynchronize(si); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, si)
    for _ in range(it): run()
    cudart.cudaEventRecord(t1, si); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / it


def test(M=4096, K=1024, N=1024):
    print(f"--- GPU MXFP8 quant (M={M} K={K} N={N}) ---")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    gx = _GpuArray(_bf(x))                                  # BF16 bits
    gfp8 = _GpuArray.zeros((M, K), np.uint8); gsc = _GpuArray.zeros((M, K // B), np.uint8)
    qms = time_ms(lambda: ct.launch(si, (M // QTM, 1, 1), _quant_mxfp8, (gx, gfp8, gsc, K // B)))

    fp8 = gfp8.to_numpy(); sc = gsc.to_numpy()
    deq = dequantize_rowblock(fp8, sc)
    rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    print(f"  quant: dequant vs bf16(x) input: {rel(deq, _bfval(x))*100:.2f}%  (FP8 quant error)")

    # end-to-end GEMM with GPU-quantized x + host-quantized w
    wf, ws = quantize_colblock(w)
    gw = _GpuArray(wf); gws = _GpuArray(ws); go = _GpuArray.zeros((M, N), np.float32)
    ct.launch(si, (M // FTM, N // FTN, 1), _fwd_mxfp8, (gfp8, gw, gsc, gws, go, M, N, K // FTK, FTM, FTN, FTK))
    cudart.cudaStreamSynchronize(si)
    y = go.to_numpy(); ref = x.astype(np.float64) @ w.astype(np.float64)
    print(f"  GPU-quant GEMM vs f32 matmul: {rel(y, ref)*100:.2f}%")

    gms = time_ms(lambda: ct.launch(si, (M // FTM, N // FTN, 1), _fwd_mxfp8, (gfp8, gw, gsc, gws, go, M, N, K // FTK, FTM, FTN, FTK)))
    print(f"  quant {qms*1000:.0f} µs | GEMM {gms*1000:.0f} µs | quant overhead {qms/(qms+gms)*100:.0f}% of fused fwd")
    for g in (gx, gfp8, gsc, gw, gws, go): g.free()
    return rel(deq, _bfval(x)) < 0.08 and rel(y, ref) < 0.05


if __name__ == "__main__":
    print("GPU MXFP8 activation quant"); print("=" * 60)
    ok = test(4096, 1024, 1024)
    ok &= test(4096, 1024, 3072)
    print("=" * 60)
    print(f"  {'PASS' if ok else 'FAIL'}")
