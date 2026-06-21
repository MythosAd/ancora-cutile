"""Call the built CUTLASS MXFP8 GEMM (cutlass_mxfp8.dll) from Python via ctypes and time it
on device buffers, vs our cuda-tile _fwd_mxfp8 at the same layer shapes. Proves the hybrid
integration path + the real callable speedup (CUTLASS warp-specialized vs our simple loop)."""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.linear import _fwd_mxfp8, mxfp8_tile, B as QB

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])

DLL = ctypes.CDLL(r"C:\project\cutlass\cutlass_mxfp8.dll")
DLL.cutlass_mxfp8_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
DLL.cutlass_mxfp8_init.restype = ctypes.c_void_p
DLL.cutlass_mxfp8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; DLL.cutlass_mxfp8_run.restype = ctypes.c_int
DLL.cutlass_mxfp8_free.argtypes = [ctypes.c_void_p]

def dalloc(nbytes):
    err, p = cdrv.cuMemAlloc(nbytes); assert err.value == 0; return p

def time_ms(fn, it=40, wm=12):
    for _ in range(wm): fn()
    cudart.cudaStreamSynchronize(SI); _, a = cudart.cudaEventCreate(); _, b = cudart.cudaEventCreate()
    cudart.cudaEventRecord(a, SI)
    for _ in range(it): fn()
    cudart.cudaEventRecord(b, SI); cudart.cudaEventSynchronize(b)
    return cudart.cudaEventElapsedTime(a, b)[1] / it


class GA:  # for the cuda-tile path
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))


def bench(M, N, K, label):
    rng = np.random.default_rng(0)
    # CUTLASS: A (M,K) fp8 rowmajor, B (K,N) fp8 colmajor, D (M,N) bf16. Values irrelevant (perf).
    dA = dalloc(M * K); cdrv.cuMemcpyHtoD(dA, rng.integers(0, 255, (M, K)).astype(np.uint8), M * K)
    dB = dalloc(K * N); cdrv.cuMemcpyHtoD(dB, rng.integers(0, 255, (K, N)).astype(np.uint8), K * N)
    dD = dalloc(M * N * 2)
    h = DLL.cutlass_mxfp8_init(M, N, K, ctypes.c_void_p(int(dA)), ctypes.c_void_p(int(dB)), ctypes.c_void_p(int(dD)))
    assert h, "init failed"
    st = DLL.cutlass_mxfp8_run(h, ctypes.c_void_p(SI)); cudart.cudaStreamSynchronize(SI)
    if st != 0:
        print(f"  {label}: CUTLASS run status={st} (nonzero=error)"); return
    run_cutlass = lambda: DLL.cutlass_mxfp8_run(h, ctypes.c_void_p(SI))

    # cuda-tile MXFP8 (ours)
    x = GA(rng.integers(0, 255, (M, K)).astype(np.uint8)); w = GA(rng.integers(0, 255, (K, N)).astype(np.uint8))
    xs = GA(np.full((M, K // QB), 127, np.uint8)); ws = GA(np.full((K // QB, N), 127, np.uint8)); o = GA.z((M, N), np.float32)
    TM, TN, TK = mxfp8_tile(N, K)
    run_ours = lambda: ct.launch(SI, (M // TM, N // TN, 1), _fwd_mxfp8, (x, w, xs, ws, o, M, N, K // TK, TM, TN, TK))

    flop = 2.0 * M * N * K
    tc, to = time_ms(run_cutlass), time_ms(run_ours)
    fc, fo = flop / (tc / 1e3) / 1e12, flop / (to / 1e3) / 1e12
    print(f"  {label:9s} M={M} N={N} K={K}:  ours {fo:5.0f} TF | CUTLASS {fc:5.0f} TF | {fc/fo:.2f}x")
    DLL.cutlass_mxfp8_free(h)
    for p in (dA, dB, dD): cdrv.cuMemFree(p)


if __name__ == "__main__":
    print("CUTLASS MXFP8 GEMM (via ctypes DLL) vs our cuda-tile GEMM"); print("=" * 64)
    bench(4096, 4096, 4096, "4096^3")
    bench(8192, 3072, 1024, "gate/up")
    bench(8192, 1024, 3072, "down")
    bench(8192, 1024, 1024, "q/o")
    bench(8192,  512, 1024, "k/v")
    print("=" * 64)
