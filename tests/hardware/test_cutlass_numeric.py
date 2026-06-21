"""Numerically-correct CUTLASS MXFP8 hybrid: quantize A (rowblock) + B (colblock, laid out
column-major), scatter our linear E8M0 scales into CUTLASS's atom layout (cutlass_mxfp8_set_scales),
run, and compare D vs the f32 reference A@B. Validates the production scale conversion."""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.quant import quantize_rowblock, quantize_colblock, dequantize_rowblock, e4m3_to_f32, B as VB

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)

DLL = ctypes.CDLL(r"C:\project\cutlass\cutlass_mxfp8.dll")
DLL.cutlass_mxfp8_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
DLL.cutlass_mxfp8_init.restype = ctypes.c_void_p
DLL.cutlass_mxfp8_set_scales.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
DLL.cutlass_mxfp8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; DLL.cutlass_mxfp8_run.restype = ctypes.c_int
DLL.cutlass_mxfp8_free.argtypes = [ctypes.c_void_p]

def dput(a):
    a = np.ascontiguousarray(a); _, p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(p, a, a.nbytes); return p, a.nbytes
def dget(p, shape, dt):
    o = np.empty(shape, dt); cdrv.cuMemcpyDtoH(o, p, o.nbytes); return o


def test(M, N, K):
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    Bm = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    # A: rowblock MXFP8 → fp8 (M,K) rowmajor + sfa (M,K/32)
    fpA, sfA = quantize_rowblock(A)                       # uint8, uint8
    # B: colblock MXFP8 (scale along K) → fp8 (K,N) rowmajor + sfb (K/32,N)
    fpB, sfB = quantize_colblock(Bm)
    fpB_cm = np.ascontiguousarray(fpB.T)                  # (N,K) rowmajor == (K,N) column-major
    sfB_t  = np.ascontiguousarray(sfB.T)                  # (N,K/32) for the scatter

    dA, _ = dput(fpA); dB, _ = dput(fpB_cm); _, dD = cdrv.cuMemAlloc(M * N * 2)
    dSFA, _ = dput(sfA); dSFB, _ = dput(sfB_t)
    h = DLL.cutlass_mxfp8_init(M, N, K, ctypes.c_void_p(int(dA)), ctypes.c_void_p(int(dB)), ctypes.c_void_p(int(dD)))
    assert h, "init failed"
    DLL.cutlass_mxfp8_set_scales(h, ctypes.c_void_p(int(dSFA)), ctypes.c_void_p(int(dSFB)), ctypes.c_void_p(SI))
    st = DLL.cutlass_mxfp8_run(h, ctypes.c_void_p(SI)); cudart.cudaStreamSynchronize(SI)
    D = b2f(dget(dD, (M, N), np.uint16))

    # references: true f32 A@B, and the "ideal MXFP8" (dequantized operands) to isolate layout vs quant error
    ref_f32 = A.astype(np.float64) @ Bm.astype(np.float64)
    Aq = dequantize_rowblock(fpA, sfA)
    Bq = (e4m3_to_f32(fpB).reshape(K // VB, VB, N) * (2.0 ** (sfB.astype(np.float32) - 127))[:, None, :]).reshape(K, N)
    ref_mxfp8 = Aq.astype(np.float64) @ Bq.astype(np.float64)
    print(f"  M={M} N={N} K={K}:  CUTLASS vs f32 ref   rms {rms(D,ref_f32)*100:.2f}%  max {rel(D,ref_f32)*100:.2f}%")
    print(f"              vs ideal-MXFP8 ref (layout-only error)  rms {rms(D,ref_mxfp8)*100:.3f}%  max {rel(D,ref_mxfp8)*100:.2f}%")
    ok = rms(D, ref_mxfp8) < 0.02   # CUTLASS output should match the dequantized-operand product closely
    print(f"              sample D[0,:3]={D[0,:3]}  ref={ref_f32[0,:3]}   {'OK' if ok else 'FAIL'}")
    DLL.cutlass_mxfp8_free(h)
    for p in (dA, dB, dD, dSFA, dSFB): cdrv.cuMemFree(p)
    return ok


if __name__ == "__main__":
    print("CUTLASS MXFP8 numeric validation (scale conversion)"); print("=" * 64)
    ok = test(512, 512, 512); ok &= test(1024, 1024, 1024); ok &= test(8192, 3072, 1024)
    print("=" * 64); print(f"  {'PASS' if ok else 'FAIL'}")
