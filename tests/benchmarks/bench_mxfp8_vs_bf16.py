"""MXFP8 (_fwd_mxfp8) vs BF16 (_gemm) at the Qwen3 layer projection shapes, training-scale
M. Decides whether swapping the forward projections to MXFP8 is worth it (and at what M)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.linear import _fwd_mxfp8, FTM, FTN, FTK, _GpuArray
from ancora.kernels.loss import _gemm, GTM, GTN, GTK
from ancora.kernels.quant import quantize_rowblock, quantize_colblock

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
_bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def time_ms(run, it=30):
    for _ in range(8): run()
    cudart.cudaStreamSynchronize(si); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, si)
    for _ in range(it): run()
    cudart.cudaEventRecord(t1, si); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / it


def cmp(M, K, N, name):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32); w = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    # MXFP8
    xf, xs = quantize_rowblock(x); wf, ws = quantize_colblock(w)
    gx, gw, gxs, gws = _GpuArray(xf), _GpuArray(wf), _GpuArray(xs), _GpuArray(ws); go = _GpuArray.zeros((M, N), np.float32)
    m8 = time_ms(lambda: ct.launch(si, (M//FTM, N//FTN, 1), _fwd_mxfp8, (gx, gw, gxs, gws, go, M, N, K//FTK, FTM, FTN, FTK)))
    # BF16
    gxb, gwb = _GpuArray(_bf(x)), _GpuArray(_bf(w)); gob = _GpuArray.zeros((M, N), np.float32)
    mb = time_ms(lambda: ct.launch(si, (M//GTM, N//GTN, 1), _gemm, (gxb, gwb, gob, K//GTK, GTM, GTN, GTK)))
    fl = 2.0 * M * N * K
    print(f"  {name:10s} M={M} K={K} N={N}:  MXFP8 {fl/(m8/1e3)/1e12:5.0f} TF | BF16 {fl/(mb/1e3)/1e12:5.0f} TF | "
          f"speedup {mb/m8:.2f}x")
    for g in (gx, gw, gxs, gws, go, gxb, gwb, gob): g.free()


if __name__ == "__main__":
    print("MXFP8 vs BF16 at Qwen3 projection shapes"); print("=" * 64)
    for M in (2048, 4096, 8192):
        cmp(M, 1024, 1024, "q/o_proj")
        cmp(M, 1024, 3072, "gate/up")
        cmp(M, 3072, 1024, "down")
        print("-" * 64)
