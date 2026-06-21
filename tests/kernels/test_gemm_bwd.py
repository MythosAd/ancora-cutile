"""Backward GEMMs (_gemm_dx = dy@Wᵀ, _gemm_dW = xᵀ@dy) vs numpy — de-risks the
device-resident backward. Keep."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _gemm_dx, _gemm_dW

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
f32bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o


def test(M=512, K=1024, N=1024, T=64):
    rng = np.random.default_rng(0)
    dy = (rng.standard_normal((M, N)) * 0.1).astype(np.float32)
    W = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    dyb, Wb, xb = bf32(f32bf(dy)), bf32(f32bf(W)), bf32(f32bf(x))

    gdy, gW, gx = GA(f32bf(dy)), GA(f32bf(W)), GA(f32bf(x))
    gdx = GA(np.zeros((M, K), np.uint16)); gdW = GA(np.zeros((K, N), np.float32))
    ct.launch(si, (M // T, K // T, 1), _gemm_dx, (gdy, gW, gdx, N // T, T, T, T))
    ct.launch(si, (K // T, N // T, 1), _gemm_dW, (gx, gdy, gdW, M // T, T, T, T))
    cudart.cudaStreamSynchronize(si)

    dx_ref = dyb.astype(np.float64) @ Wb.T.astype(np.float64)
    dW_ref = xb.T.astype(np.float64) @ dyb.astype(np.float64)
    rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    edx, edW = rel(bf32(gdx.np()), dx_ref), rel(gdW.np(), dW_ref)
    ok = edx < 0.02 and edW < 0.02
    print(f"  M={M} K={K} N={N}: dx=dy@Wᵀ {edx*100:.2f}% | dW=xᵀ@dy {edW*100:.2f}%  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("backward GEMMs"); print("=" * 50)
    ok = test(512, 1024, 1024); ok &= test(256, 1024, 3072)
    print("=" * 50); print(f"  {'PASS' if ok else 'FAIL'}")
