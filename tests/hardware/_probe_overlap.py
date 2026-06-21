"""DECISIVE feasibility probe for attention/GEMM OVERLAP (the path from ~55%→80% MFU).

The megakernel overlap (HazyResearch) fills attention's idle tensor cores (head_dim=64 ⇒
~37% TC util) with a neighbor GEMM's matmul. True overlap = INTRA-SM warp interleaving =
persistent kernel — which cuda-tile 1.4.0 can't express (num_worker_warps no-op, no warp
spec, no persistent loop). The only cuda-tile-native overlap is MULTI-STREAM concurrency
(block-level). This probes whether that actually overlaps:

  T_seq  = attn + gemm on ONE stream (sequential)
  T_conc = attn on stream A, gemm on stream B, sync both
  overlap iff T_conc < T_seq (ideally ≈ max(T_attn, T_gemm)).

Tested at SMALL grids (room to co-reside) and the REAL layer size (each saturates SMs)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.attention import _attn_fwd, BQ, D as DH
from ancora.kernels.fused import _gemm_bf16
from ancora.kernels.loss import GTM, GTN, GTK

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
sA = dev.create_stream(); sB = dev.create_stream()
siA = int(sA.__cuda_stream__()[1]); siB = int(sB.__cuda_stream__()[1])
_bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

def time_ms(fn, it=60, wm=15):
    # wall-clock + full-device sync: correctly captures work on ANY stream (events on a
    # single stream miss cross-stream concurrency). it large so compute >> host launch.
    for _ in range(wm): fn()
    cudart.cudaDeviceSynchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    cudart.cudaDeviceSynchronize()
    return (time.perf_counter() - t) / it * 1000


def probe(B, S, label, Hq=16, Hkv=8, Dh=64, H=1024, gemm_N=3072):
    rng = np.random.default_rng(0); M = B * S; NQB = S // BQ
    gQ = GA(_bf(rng.standard_normal((B*Hq*S, Dh)).astype(np.float32)))
    gK = GA(_bf(rng.standard_normal((B*Hkv*S, Dh)).astype(np.float32)))
    gV = GA(_bf(rng.standard_normal((B*Hkv*S, Dh)).astype(np.float32)))
    gO = GA.z((B*Hq*S, Dh), np.float32); gL = GA.z((B*Hq*S, 1), np.float32)
    gA = GA(_bf(rng.standard_normal((M, H)).astype(np.float32)))
    gW = GA(_bf(rng.standard_normal((H, gemm_N)).astype(np.float32)))
    gC = GA.z((M, gemm_N), np.uint16)
    scale = 1.0 / math.sqrt(Dh)
    attn = lambda s=siA: ct.launch(s, (NQB, B*Hq, 1), _attn_fwd, (gQ, gK, gV, gO, gL, NQB, NQB, Hq, Hkv, scale))
    gemm = lambda s=siA: ct.launch(s, (M//GTM, gemm_N//GTN, 1), _gemm_bf16, (gA, gW, gC, H//GTK, GTM, GTN, GTK))
    t_a = time_ms(lambda: attn(siA)); t_g = time_ms(lambda: gemm(siA))
    t_seq = time_ms(lambda: (attn(siA), gemm(siA)))
    t_conc = time_ms(lambda: (attn(siA), gemm(siB)))
    ov = (t_seq - t_conc) / max(t_a, t_g) * 100   # % of the smaller op hidden
    print(f"  {label}: attn {t_a*1000:.0f}us  gemm {t_g*1000:.0f}us  | seq {t_seq*1000:.0f}us  "
          f"concurrent {t_conc*1000:.0f}us  → {'OVERLAP '+f'{ov:.0f}%' if t_conc < t_seq*0.97 else 'NO overlap'}")


if __name__ == "__main__":
    print("Attention/GEMM overlap feasibility (multi-stream, cuda-tile)"); print("=" * 70)
    probe(1, 256, "small  B1 S256 ")   # tiny grids — room to co-reside
    probe(1, 512, "medium B1 S512 ")
    probe(4, 2048, "real   B4 S2048")  # each saturates SMs
    print("=" * 70)
    print("  If NO overlap even at small sizes → cuda-tile can't overlap; 80% needs a")
    print("  hand-written persistent megakernel (nvcc/PTX). If overlap at small only →")
    print("  limited by SM saturation; marginal for training-size forward.")
