"""
Autotune the resident-path BF16 GEMM (_gemm_bf16) at the Qwen3-layer's REAL shapes,
not 4096^3. The layer MFU is GEMM-bound at ~25% of BF16 peak (bench_layer_mfu); the
4096^3-tuned config (TM128/TN256/TK32) need not be optimal for M=8192, N/K∈{512,1024,3072}.

Sweeps (TM,TN,TK,occupancy) via ct.tune.exhaustive_search at each layer GEMM shape and
prints the best config + TFLOPS. Constraint: MMA output rows TM<=128 (CLAUDE pitfall 0b).
Keep — re-run whenever cuda-tile updates.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _gemm_bf16

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
PEAK_BF16 = 165.0

def _bf(x): return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
    def free(s): cdrv.cuMemFree(s.p)

# candidate tiles (TM<=128 hard limit; TN/TK any divisor); occupancy 1..3 (>=4 worse per CLAUDE)
TMS = [64, 128]; TNS = [64, 128, 256]; TKS = [32, 64, 128]; OCCS = [1, 2, 3]

def tune_shape(M, N, K, label):
    rng = np.random.default_rng(0)
    A = GA(_bf(rng.standard_normal((M, K)).astype(np.float32)))
    Bm = GA(_bf(rng.standard_normal((K, N)).astype(np.float32)))
    C = GA.z((M, N), np.uint16)
    flop = 2.0 * M * N * K

    space = [(tm, tn, tk, occ)
             for tm in TMS for tn in TNS for tk in TKS for occ in OCCS
             if M % tm == 0 and N % tn == 0 and K % tk == 0]
    grid_fn = lambda c: (M // c[0], N // c[1], 1)
    args_fn = lambda c: (A, Bm, C, K // c[2], c[0], c[1], c[2])
    hints_fn = lambda c: {"occupancy": c[3]}

    res = ct.tune.exhaustive_search(space, SI, grid_fn, _gemm_bf16, args_fn, hints_fn, quiet=True)
    b = res.best
    tf = flop / (b.mean_us * 1e-6) / 1e12
    print(f"{label:18s} M={M} N={N} K={K}:  best {b.config}  {b.mean_us:7.1f}us  "
          f"{tf:6.1f} TFLOPS ({tf/PEAK_BF16*100:.0f}% peak)   [{len(res.successes)}/{len(space)} ok]")
    # show top-5
    top = sorted(res.successes, key=lambda m: m.mean_us)[:5]
    for m in top:
        t = flop / (m.mean_us * 1e-6) / 1e12
        print(f"      {str(m.config):26s} {m.mean_us:7.1f}us  {t:6.1f} TFLOPS")
    for g in (A, Bm, C): g.free()
    return b.config, tf

if __name__ == "__main__":
    print("Autotune _gemm_bf16 at Qwen3-layer GEMM shapes (M=B*S=8192)"); print("=" * 78)
    tune_shape(8192, 3072, 1024, "gate/up (N=3072)")
    tune_shape(8192, 1024, 3072, "down (K=3072)")
    tune_shape(8192, 1024, 1024, "q/o (1024^2)")
    tune_shape(8192,  512, 1024, "k/v (N=512)")
    print("=" * 78)
