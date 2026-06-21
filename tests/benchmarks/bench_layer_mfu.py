"""
Forward TransformerLayer MFU — kernel-only (CUDA-event timed, pre-uploaded device
buffers, NO host glue/transfers). Answers: are the forward kernels efficient enough?

Counts model FLOPs (projections + causal attention), times each compute kernel at the
LAYER's real shapes, reports per-category TFLOPS and overall MFU. The host
reshape/transpose + f32->bf16 casts between kernels are EXCLUDED — they are memory ops
the megakernel fuses away; this isolates compute efficiency.

Peaks (RTX 5080 Laptop sm_120a, corrected estimates): BF16 ~165 TFLOPS, MXFP8 ~330.
The projections here run BF16 (loss._gemm). NOTE the headline "174 TFLOPS linear" is
MXFP8 at 4096^3 — the layer's smaller BF16 GEMMs will sit lower; that gap is the point.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

cudart.cudaFree(0); _dev = cc.Device(0); _dev.set_current()
_so = _dev.create_stream(); SI = int(_so.__cuda_stream__()[1])
PEAK_BF16 = 165.0   # TFLOPS, approximate sustained BF16 tensor-core peak

def _bf(x): return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
    def free(s): cdrv.cuMemFree(s.p)

def time_ms(launch, iters=50, warmup=10):
    for _ in range(warmup): launch()
    _so.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, _so.__cuda_stream__()[1])
    for _ in range(iters): launch()
    cudart.cudaEventRecord(t1, _so.__cuda_stream__()[1]); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / iters


def bench(B, S, H=1024, Hq=16, Hkv=8, Dh=64, I=3072):
    from ancora.kernels.loss import _gemm, GTM, GTN, GTK
    from ancora.kernels.attention import _attn_fwd, D, BQ
    from ancora.kernels.norm import _rmsnorm_stats, _rmsnorm_apply, TM as NTM, TH
    from ancora.kernels.rope import _rope_fwd, RTM
    from ancora.kernels.activation import _swiglu_fwd, TM as STM, TI
    M = B * S; qd, kd = Hq * Dh, Hkv * Dh
    rng = np.random.default_rng(0)
    def W(k, n): return GA(_bf(rng.standard_normal((k, n)).astype(np.float32)))
    def A(m, n): return GA(_bf(rng.standard_normal((m, n)).astype(np.float32)))

    # ---- buffers (correct shapes/dtypes; values irrelevant for timing) ----
    gh   = A(M, H)                                   # rmsnorm-ed hidden (bf16) feeding GEMMs
    wq, wk, wv = W(H, qd), W(H, kd), W(H, kd)
    wo   = W(qd, H); wg, wu = W(H, I), W(H, I); wd = W(I, H)
    cq   = GA.z((M, qd), np.float32); ck = GA.z((M, kd), np.float32); cv = GA.z((M, kd), np.float32)
    co   = GA.z((M, H), np.float32); cg = GA.z((M, I), np.float32); cu = GA.z((M, I), np.float32); cd = GA.z((M, H), np.float32)
    # attention buffers (head-major bf16)
    gQ, gK, gV = A(B*Hq*S, Dh), A(B*Hkv*S, Dh), A(B*Hkv*S, Dh)
    gO = GA.z((B*Hq*S, Dh), np.float32); gL = GA.z((B*Hq*S, 1), np.float32)
    # norm / rope / swiglu buffers
    gx = A(M, H); gw1 = A(1, H); gy = GA.z((M, H), np.uint16); gr = GA.z((M, 1), np.float32)
    gg = A(M, I); gu2 = A(M, I); gsw = GA.z((M, I), np.uint16)
    rc, rs = GA(np.zeros((S, Dh//2), np.float32)), GA(np.zeros((S, Dh//2), np.float32))
    grp = A(B*Hq*S, Dh); gry = GA.z((B*Hq*S, Dh), np.uint16)

    G = lambda A_, Bm, C, K, N: ct.launch(SI, (M//GTM, N//GTN, 1), _gemm, (A_, Bm, C, K//GTK, GTM, GTN, GTK))
    def gemms():   # all 7 projections
        G(gh, wq, cq, H, qd); G(gh, wk, ck, H, kd); G(gh, wv, cv, H, kd)
        G(gh, wo, co, qd if qd==H else H, H)        # o: (M,qd)@(qd,H); here qd==H
        G(gh, wg, cg, H, I);  G(gh, wu, cu, H, I);  G(gh, wd, cd, I, H)
    def attn():
        ct.launch(SI, (S//BQ, B*Hq, 1), _attn_fwd, (gQ, gK, gV, gO, gL, S//BQ, S//BQ, Hq, Hkv, 1/math.sqrt(Dh)))
    def norms():   # 4 RMSNorm: input_ln, post_ln (H=1024) + q_norm,k_norm (rows*=heads, H=Dh)
        for (rows, hh) in [(M, H), (M, H)]:
            ct.launch(SI, (rows//NTM, 1, 1), _rmsnorm_stats, (gx, gr, hh//TH, 1.0/hh, 1e-6))
            ct.launch(SI, (rows//NTM, 1, 1), _rmsnorm_apply, (gx, gw1, gr, gy, hh//TH))
    def rope():
        for _ in range(2):
            ct.launch(SI, (S//RTM, B*Hq, 1), _rope_fwd, (grp, rc, rs, gry, S//RTM, Dh//2))
    def swiglu():
        ct.launch(SI, (M//STM, I//TI, 1), _swiglu_fwd, (gg, gu2, gsw))

    t_g, t_a, t_n, t_r, t_s = (time_ms(f) for f in (gemms, attn, norms, rope, swiglu))
    total = t_g + t_a + t_n + t_r + t_s

    gemm_fl = 2.0*M*(H*qd + 2*H*kd + qd*H + 3*H*I)
    attn_fl = 2.0*B*Hq*S*S*Dh                 # causal (~half of 4*B*Hq*S^2*Dh)
    flops   = gemm_fl + attn_fl
    eff = flops/(total/1e3)/1e12
    print(f"B={B} S={S} M={M}:")
    print(f"  GEMMs  {t_g:.3f} ms  {gemm_fl/(t_g/1e3)/1e12:5.0f} TFLOPS  ({gemm_fl/(t_g/1e3)/1e12/PEAK_BF16*100:.0f}% BF16 peak)")
    print(f"  Attn   {t_a:.3f} ms  {attn_fl/(t_a/1e3)/1e12:5.0f} TFLOPS")
    print(f"  norm   {t_n:.3f} ms |  rope {t_r:.3f} ms |  swiglu {t_s:.3f} ms   (memory-bound, ~0 FLOPs)")
    print(f"  TOTAL  {total:.3f} ms  → {eff:5.0f} TFLOPS effective  = {eff/PEAK_BF16*100:.0f}% MFU (vs {PEAK_BF16:.0f} BF16 peak)")
    print(f"  time split: GEMM {t_g/total*100:.0f}% | attn {t_a/total*100:.0f}% | norm {t_n/total*100:.0f}% | rope {t_r/total*100:.0f}% | swiglu {t_s/total*100:.0f}%")
    for g in (gh,wq,wk,wv,wo,wg,wu,wd,cq,ck,cv,co,cg,cu,cd,gQ,gK,gV,gO,gL,gx,gw1,gy,gr,gg,gu2,gsw,rc,rs,grp,gry): g.free()


if __name__ == "__main__":
    print(f"Qwen3 layer FORWARD MFU — kernel-only, BF16 projections, head_dim=64")
    print("=" * 70)
    bench(1, 2048)
    print("-" * 70)
    bench(4, 2048)
    print("=" * 70)
    print("  (projections=BF16 loss._gemm; MXFP8 fwd path would ~2x the GEMM TFLOPS)")
