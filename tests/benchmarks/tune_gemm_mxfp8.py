"""
Autotune the MXFP8 forward GEMM (_fwd_mxfp8, ct.mma_scaled) at the Qwen3-layer's REAL
shapes. BF16 plateaus at ~40% of peak (tune_gemm_bf16); MXFP8 (peak ~330) is the lever
for MFU>80%. Confirms the achievable MXFP8 TFLOPS per layer GEMM shape + best tile.

mma_scaled needs TK % 32 == 0 (B=32 scale block). x(M,K) fp8, w(K,N) fp8 pre-transposed,
scales E8M0. Keep — re-run on cuda-tile updates.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.linear import _fwd_mxfp8, B

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
PEAK_BF16 = 165.0; PEAK_MXFP8 = 330.0

class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
    def free(s): cdrv.cuMemFree(s.p)

TMS = [64, 128]; TNS = [64, 128, 256]; TKS = [32, 64, 128, 256]; OCCS = [1, 2, 3]

def tune_shape(M, N, K, label):
    rng = np.random.default_rng(0)
    x = GA((rng.integers(0, 255, (M, K))).astype(np.uint8))
    w = GA((rng.integers(0, 255, (K, N))).astype(np.uint8))
    xs = GA(np.full((M, K // B), 127, np.uint8))   # E8M0 1.0
    ws = GA(np.full((K // B, N), 127, np.uint8))
    out = GA.z((M, N), np.float32)
    flop = 2.0 * M * N * K

    space = [(tm, tn, tk, occ)
             for tm in TMS for tn in TNS for tk in TKS for occ in OCCS
             if M % tm == 0 and N % tn == 0 and K % tk == 0]
    grid_fn = lambda c: (M // c[0], N // c[1], 1)
    args_fn = lambda c: (x, w, xs, ws, out, M, N, K // c[2], c[0], c[1], c[2])
    hints_fn = lambda c: {"occupancy": c[3]}
    res = ct.tune.exhaustive_search(space, SI, grid_fn, _fwd_mxfp8, args_fn, hints_fn, quiet=True)
    b = res.best
    tf = flop / (b.mean_us * 1e-6) / 1e12
    print(f"{label:18s} M={M} N={N} K={K}:  best {b.config}  {b.mean_us:7.1f}us  "
          f"{tf:6.1f} TFLOPS ({tf/PEAK_MXFP8*100:.0f}% mxfp8 / {tf/PEAK_BF16*100:.0f}% bf16)   [{len(res.successes)}/{len(space)}]")
    for m in sorted(res.successes, key=lambda z: z.mean_us)[:5]:
        t = flop / (m.mean_us * 1e-6) / 1e12
        print(f"      {str(m.config):26s} {m.mean_us:7.1f}us  {t:6.1f} TFLOPS")
    for g in (x, w, xs, ws, out): g.free()

if __name__ == "__main__":
    print("Autotune _fwd_mxfp8 at Qwen3-layer GEMM shapes (M=8192)"); print("=" * 80)
    tune_shape(8192, 3072, 1024, "gate/up (N=3072)")
    tune_shape(8192, 1024, 3072, "down (K=3072)")
    tune_shape(8192, 1024, 1024, "q/o (1024^2)")
    tune_shape(8192,  512, 1024, "k/v (N=512)")
    print("=" * 80)
