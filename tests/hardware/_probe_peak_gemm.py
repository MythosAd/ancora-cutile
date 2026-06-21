"""Resolve the real BF16/MXFP8 ceiling on this GPU with LARGE compute-bound GEMMs (4096³,
deep K → MMA-bound, real memory path). The in-register MMA microbench (_probe_peak) gave
BF16=81/MXFP8=323 (4× ratio, suspicious vs the 2× spec ratio). A real 4096³ GEMM is the
honest achievable number to normalize MFU against."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _gemm_bf16
from ancora.kernels.linear import _fwd_mxfp8, B as MXB
cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])

class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

def tmms(fn, it=20, wm=6):
    for _ in range(wm): fn()
    cudart.cudaStreamSynchronize(SI); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, SI)
    for _ in range(it): fn()
    cudart.cudaEventRecord(t1, SI); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / it

def bf16_best(M, N, K):
    bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    rng = np.random.default_rng(0)
    A = GA(bf(rng.standard_normal((M, K)).astype(np.float32))); Bm = GA(bf(rng.standard_normal((K, N)).astype(np.float32)))
    C = GA.z((M, N), np.uint16); flop = 2.0 * M * N * K; best = 0; bcfg = None
    for TM, TN, TK in [(128, 128, 64), (128, 256, 32), (64, 256, 64), (128, 128, 32), (64, 128, 128), (128, 64, 64)]:
        if M % TM or N % TN or K % TK: continue
        try:
            t = tmms(lambda TM=TM, TN=TN, TK=TK: ct.launch(SI, (M // TM, N // TN, 1), _gemm_bf16, (A, Bm, C, K // TK, TM, TN, TK)))
            tf = flop / (t / 1e3) / 1e12
            if tf > best: best, bcfg = tf, (TM, TN, TK)
        except Exception: pass
    return best, bcfg

def mxfp8_best(M, N, K):
    rng = np.random.default_rng(0)
    x = GA(rng.integers(0, 255, (M, K)).astype(np.uint8)); w = GA(rng.integers(0, 255, (K, N)).astype(np.uint8))
    xs = GA(np.full((M, K // MXB), 127, np.uint8)); ws = GA(np.full((K // MXB, N), 127, np.uint8))
    out = GA.z((M, N), np.float32); flop = 2.0 * M * N * K; best = 0; bcfg = None
    for TM, TN, TK in [(128, 256, 32), (128, 64, 128), (128, 64, 256), (64, 128, 128), (128, 128, 64), (64, 256, 128)]:
        if M % TM or N % TN or K % TK: continue
        try:
            t = tmms(lambda TM=TM, TN=TN, TK=TK: ct.launch(SI, (M // TM, N // TN, 1), _fwd_mxfp8, (x, w, xs, ws, out, M, N, K // TK, TM, TN, TK)))
            tf = flop / (t / 1e3) / 1e12
            if tf > best: best, bcfg = tf, (TM, TN, TK)
        except Exception: pass
    return best, bcfg

if __name__ == "__main__":
    print("Large compute-bound GEMM ceiling (4096^3) — the real achievable peak"); print("=" * 64)
    bf, bc = bf16_best(4096, 4096, 4096); print(f"  BF16  4096^3: {bf:6.0f} TFLOPS  (best {bc})")
    mf, mc = mxfp8_best(4096, 4096, 4096); print(f"  MXFP8 4096^3: {mf:6.0f} TFLOPS  (best {mc})")
    print("=" * 64)
    print(f"  in-register microbench said BF16 81 / MXFP8 323. If 4096^3 BF16 << MXFP8/2,")
    print(f"  the consumer BF16+FP32acc throttle is real → MFU vs ~{bf:.0f} BF16 not 165.")
