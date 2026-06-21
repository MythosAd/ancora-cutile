"""STEP 3: numerically validate the CUTLASS NVFP4 GEMM (cutlass_nvfp4.dll). Quantize A (rowblock)
+ Bᵀ (→ column-major (K,N)), scatter linear E4M3 scales into the atom layout, run, compare D vs
f32 ref (≈ FP4 quant error ~10%) and vs the ideal-NVFP4 dequant product (layout-only error, ~0%)."""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.quant_nvfp4 import quantize_nvfp4_rowblock, dequantize_nvfp4

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)

D = ctypes.CDLL(r"C:\project\cutlass\cutlass_nvfp4.dll")
D.cutlass_nvfp4_init.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3; D.cutlass_nvfp4_init.restype = ctypes.c_void_p
D.cutlass_nvfp4_set_scales.argtypes = [ctypes.c_void_p] * 4
D.cutlass_nvfp4_run.argtypes = [ctypes.c_void_p] * 2; D.cutlass_nvfp4_run.restype = ctypes.c_int
D.cutlass_nvfp4_free.argtypes = [ctypes.c_void_p]

def dput(a):
    a = np.ascontiguousarray(a); _, p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(p, a, a.nbytes); return p
def dget(p, shape, dt):
    o = np.empty(shape, dt); cdrv.cuMemcpyDtoH(o, p, o.nbytes); return o


def test(M, N, K):
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    Bm = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    fpA, sfA = quantize_nvfp4_rowblock(A)                 # (M,K/2), (M,K/16)
    fpBt, sfBt = quantize_nvfp4_rowblock(Bm.T.copy())     # quantize Bᵀ (N,K) → (N,K/2)==(K,N) col-major, (N,K/16)

    dA, dB, dD = dput(fpA), dput(fpBt), cdrv.cuMemAlloc(M * N * 2)[1]
    dSFA, dSFB = dput(sfA), dput(sfBt)
    h = D.cutlass_nvfp4_init(M, N, K, ctypes.c_void_p(int(dA)), ctypes.c_void_p(int(dB)), ctypes.c_void_p(int(dD)))
    assert h, "init failed"
    D.cutlass_nvfp4_set_scales(h, ctypes.c_void_p(int(dSFA)), ctypes.c_void_p(int(dSFB)), ctypes.c_void_p(SI))
    st = D.cutlass_nvfp4_run(h, ctypes.c_void_p(SI)); cudart.cudaStreamSynchronize(SI)
    Dout = b2f(dget(dD, (M, N), np.uint16))

    ref_f32 = A.astype(np.float64) @ Bm.astype(np.float64)
    Aq = dequantize_nvfp4(fpA, sfA); Bq = dequantize_nvfp4(fpBt, sfBt).T   # ideal NVFP4 operands
    ref_nv = Aq.astype(np.float64) @ Bq.astype(np.float64)
    print(f"  M={M} N={N} K={K}: NVFP4 vs f32  rms {rms(Dout,ref_f32)*100:.2f}%  | vs ideal-NVFP4 (layout) rms {rms(Dout,ref_nv)*100:.3f}%")
    ok = (st == 0) and rms(Dout, ref_nv) < 0.03
    print(f"             sample D[0,:3]={Dout[0,:3]}  ref={ref_f32[0,:3]}   {'OK' if ok else 'FAIL'}")
    D.cutlass_nvfp4_free(h)
    for p in (dA, dB, dD, dSFA, dSFB): cdrv.cuMemFree(p)
    return ok


if __name__ == "__main__":
    print("CUTLASS NVFP4 GEMM numeric validation — step 3"); print("=" * 64)
    ok = test(512, 512, 512); ok &= test(1024, 1024, 1024); ok &= test(8192, 3072, 1024)
    print("=" * 64); print(f"  {'PASS' if ok else 'FAIL'}")
