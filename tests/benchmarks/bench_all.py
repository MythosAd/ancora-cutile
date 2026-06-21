"""
ANCORA kernel performance suite — run after every cuda-tile / toolkit upgrade to
track throughput (regressions AND improvements; attention is still "experimental").

Measures kernel-only time (CUDA events, pre-uploaded data — no host overhead) for:
  linear MXFP8 GEMM, attention fwd, attention bwd (dQ/dK/dV), fused linear-CE fwd+bwd.

Baseline recorded 2026-05-31 (cuda-tile 1.4.0, RTX 5080 Laptop sm_120a):
  COMPUTE-bound (TFLOPS):
    linear MXFP8 4096³        : ~174 TFLOPS
    attention fwd  S=2048     : ~55 TFLOPS  (causal, GQA, head_dim=64)
    attention bwd  S=2048     : ~36 TFLOPS
    linear-CE fwd+bwd 4096³V8k: ~59 TFLOPS
  MEMORY-bound (GB/s effective, peak 896; 200-iter sustained):
    RMSNorm fwd/bwd M=4096 H=1024 : ~455 / ~665 GB/s  (51% / 74%)
      (fwd 51% = 2-kernel stats+apply moves 1.5x naive traffic; ~72% on real bytes)
    RoPE   fwd/bwd  S=2048 D=128  : ~727 / ~748 GB/s  (81% / 83%)
  NOTE: warm/cache-hot single-shot runs read ~10-15% higher (clock boost); 200-iter
  sustained is the honest steady-state. Standalone norm/RoPE are launch+L2 bound at
  this size — they fuse into GEMM epilogues in the megakernel (CODA), so the
  end-to-end cost is hidden. Run after every toolkit upgrade to catch regressions.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

cudart.cudaFree(0); _dev = cc.Device(0); _dev.set_current()
_so = _dev.create_stream(); SI = int(_so.__cuda_stream__()[1])

def _bf(x): return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
    def free(s): cdrv.cuMemFree(s.p)

def time_ms(launch, iters=30, warmup=5):
    for _ in range(warmup): launch()
    _so.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, _so.__cuda_stream__()[1])
    for _ in range(iters): launch()
    cudart.cudaEventRecord(t1, _so.__cuda_stream__()[1]); cudart.cudaEventSynchronize(t1)
    _, ms = cudart.cudaEventElapsedTime(t0, t1)
    return ms / iters


def bench_linear():
    from ancora.kernels.linear import _fwd_mxfp8, FTM, FTN, FTK, B
    M = N = K = 4096
    x  = GA(np.full((M, K),      0x38, np.uint8)); w = GA(np.full((K, N), 0x38, np.uint8))
    xs = GA(np.full((M, K // B), 0x7F, np.uint8)); ws = GA(np.full((K // B, N), 0x7F, np.uint8))
    o  = GA.z((M, N), np.float32); KB = K // FTK
    ms = time_ms(lambda: ct.launch(SI, (M//FTM, N//FTN, 1), _fwd_mxfp8, (x, w, xs, ws, o, M, N, KB, FTM, FTN, FTK)))
    for g in (x, w, xs, ws, o): g.free()
    return f"linear MXFP8 GEMM  4096³            {ms:.3f} ms  {2.0*M*N*K/(ms/1e3)/1e12:5.0f} TFLOPS"


def bench_attn_fwd(B=1, H=16, Hkv=8, S=2048):
    from ancora.kernels.attention import _attn_fwd, D, BQ
    NQB = NKVB = S // BQ; sc = 1/math.sqrt(D); rng = np.random.default_rng(0)
    mk = lambda h: GA(_bf(rng.standard_normal((B*h*S, D)).astype(np.float32)))
    gQ, gK, gV = mk(H), mk(Hkv), mk(Hkv); gO = GA.z((B*H*S, D), np.float32); gL = GA.z((B*H*S, 1), np.float32)
    ms = time_ms(lambda: ct.launch(SI, (NQB, B*H, 1), _attn_fwd, (gQ, gK, gV, gO, gL, NQB, NKVB, H, Hkv, sc)))
    tf = 2.0*2.0*B*H*S*S*D*0.5/(ms/1e3)/1e12
    for g in (gQ, gK, gV, gO, gL): g.free()
    return f"attention fwd  causal GQA S={S}    {ms:.3f} ms  {tf:5.0f} TFLOPS"


def bench_attn_bwd(B=1, H=16, Hkv=8, S=2048):
    from ancora.kernels.attention import _attn_bwd_dq, _attn_bwd_dkdv, D, BQ
    G = H // Hkv; NQB = NKVB = S // BQ; sc = 1/math.sqrt(D); rng = np.random.default_rng(0)
    mk = lambda h: GA(_bf(rng.standard_normal((B*h*S, D)).astype(np.float32)))
    gQ, gK, gV, gdO = mk(H), mk(Hkv), mk(Hkv), mk(H)
    gL = GA.z((B*H*S, 1), np.float32); gDl = GA.z((B*H*S, 1), np.float32)
    gdQ = GA.z((B*H*S, D), np.float32); gdK = GA.z((B*Hkv*S, D), np.float32); gdV = GA.z((B*Hkv*S, D), np.float32)
    def L():
        ct.launch(SI, (NQB, B*H, 1), _attn_bwd_dq, (gQ, gK, gV, gdO, gL, gDl, gdQ, NQB, NKVB, H, Hkv, sc))
        ct.launch(SI, (NKVB, B*Hkv, 1), _attn_bwd_dkdv, (gQ, gK, gV, gdO, gL, gDl, gdK, gdV, NQB, NKVB, H, Hkv, G, sc))
    ms = time_ms(L)
    tf = 2.5*2.0*2.0*B*H*S*S*D*0.5/(ms/1e3)/1e12   # bwd ≈ 2.5× fwd FLOPs
    for g in (gQ, gK, gV, gdO, gL, gDl, gdQ, gdK, gdV): g.free()
    return f"attention bwd  dQ/dK/dV S={S}      {ms:.3f} ms  {tf:5.0f} TFLOPS"


def bench_rmsnorm(M=4096, H=1024):
    """Memory-bound → GB/s (% of 896 GB/s peak). fwd traffic ~4MH (read x + write y);
    bwd ~10MH (dx: read x,dy + write dx = 6MH; dw: read x,dy = 4MH)."""
    from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_bwd_dx,
                                      _rmsnorm_dw_part, _rmsnorm_dw_reduce, TM, TH, TD, PART)
    rng = np.random.default_rng(0)
    x = _bf(rng.standard_normal((M, H)).astype(np.float32)); w = _bf(rng.standard_normal((1, H)).astype(np.float32))
    gx = GA(x); gw = GA(w); gy = GA.z((M, H), np.uint16); gr = GA.z((M, 1), np.float32)
    gdy = GA(x); gdx = GA.z((M, H), np.uint16)
    gpart = GA.z((PART, H), np.float32); gdw = GA.z((1, H), np.float32)
    iH = 1.0 / H; MB = M // TM; BPP = (MB + PART - 1) // PART
    def fwd():
        ct.launch(SI, (M//TM, 1, 1), _rmsnorm_stats, (gx, gr, H//TH, iH, 1e-6))
        ct.launch(SI, (M//TM, 1, 1), _rmsnorm_apply, (gx, gw, gr, gy, H//TH))
    fms = time_ms(fwd, iters=200)
    def bwd():
        ct.launch(SI, (M//TM, 1, 1), _rmsnorm_bwd_dx, (gx, gw, gdy, gr, gdx, H//TH, iH))
        ct.launch(SI, (H//TD, PART, 1), _rmsnorm_dw_part, (gx, gdy, gr, gpart, MB, BPP))
        ct.launch(SI, (H//TD, 1, 1), _rmsnorm_dw_reduce, (gpart, gdw))
    bms = time_ms(bwd, iters=200)
    # Effective bandwidth vs the MINIMAL DRAM a naive impl moves (re-reads hit L2):
    fgb = 4.0*M*H / (fms/1e3) / 1e9   # naive fwd = read x + write y (our 2-kernel adds an L2 x re-read)
    bgb = 10.0*M*H / (bms/1e3) / 1e9  # naive bwd = dx(read x,dy,w + write dx) + dw(read x,dy)
    for g in (gx, gw, gy, gr, gdy, gdx, gpart, gdw): g.free()
    return (f"RMSNorm fwd    M={M} H={H}      {fms:.3f} ms  {fgb:4.0f} GB/s ({fgb/896*100:.0f}%)\n"
            f"  RMSNorm bwd    M={M} H={H}      {bms:.3f} ms  {bgb:4.0f} GB/s ({bgb/896*100:.0f}%)")


def bench_rope(Bb=1, Hh=16, S=2048, D=128):
    """Memory-bound → GB/s. fwd/bwd each read x + write y = 4·R·D bytes (R=B·H·S)."""
    from ancora.kernels.rope import _rope_fwd, _rope_bwd, build_cos_sin, RTM
    rng = np.random.default_rng(0); R = Bb*Hh*S
    x = _bf(rng.standard_normal((R, D)).astype(np.float32))
    cos, sin = build_cos_sin(S, D, 1e6)
    gx = GA(x); gc = GA(cos); gs = GA(sin); gy = GA.z((R, D), np.uint16); gdx = GA.z((R, D), np.uint16)
    Dh = D // 2; NSB = S // RTM
    fms = time_ms(lambda: ct.launch(SI, (NSB, Bb*Hh, 1), _rope_fwd, (gx, gc, gs, gy, NSB, Dh)), iters=200)
    bms = time_ms(lambda: ct.launch(SI, (NSB, Bb*Hh, 1), _rope_bwd, (gx, gc, gs, gdx, NSB, Dh)), iters=200)
    fgb = 4.0*R*D / (fms/1e3) / 1e9
    bgb = 4.0*R*D / (bms/1e3) / 1e9
    for g in (gx, gc, gs, gy, gdx): g.free()
    return (f"RoPE fwd       B={Bb} H={Hh} S={S} D={D}  {fms:.3f} ms  {fgb:4.0f} GB/s ({fgb/896*100:.0f}%)\n"
            f"  RoPE bwd       B={Bb} H={Hh} S={S} D={D}  {bms:.3f} ms  {bgb:4.0f} GB/s ({bgb/896*100:.0f}%)")


def bench_linear_ce(M=4096, H=1024, V=8192):
    from ancora.kernels.loss import _gemm, _ce_stats, _ce_grad, GTM, GTN, GTK, CTM, TV
    rng = np.random.default_rng(1)
    hid = (rng.standard_normal((M, H))*0.3).astype(np.float32); w = (rng.standard_normal((H, V))*0.3).astype(np.float32)
    gh = GA(_bf(hid)); gw = GA(_bf(w)); glab = GA(rng.integers(0, V, (M, 1)).astype(np.int32))
    ga = GA(rng.standard_normal((M, 1)).astype(np.float32)); gLg = GA.z((M, V), np.float32)
    gp = GA.z((M, 1), np.float32); gls = GA.z((M, 1), np.float32); gG = GA.z((M, V), np.uint16)
    gwT = GA(_bf(np.ascontiguousarray(w.T))); ghT = GA(_bf(np.ascontiguousarray(hid.T)))
    gdh = GA.z((M, H), np.float32); gdw = GA.z((H, V), np.float32)
    def L():
        ct.launch(SI, (M//GTM, V//GTN, 1), _gemm, (gh, gw, gLg, H//GTK, GTM, GTN, GTK))
        ct.launch(SI, (M//CTM, 1, 1), _ce_stats, (gLg, glab, gp, gls, V//TV))
        ct.launch(SI, (M//CTM, 1, 1), _ce_grad, (gLg, gls, glab, ga, gG, V//TV, 1.0/M))
        ct.launch(SI, (M//GTM, H//GTN, 1), _gemm, (gG, gwT, gdh, V//GTK, GTM, GTN, GTK))
        ct.launch(SI, (H//GTM, V//GTN, 1), _gemm, (ghT, gG, gdw, M//GTK, GTM, GTN, GTK))
    ms = time_ms(L, iters=20)
    for g in (gh, gw, glab, ga, gLg, gp, gls, gG, gwT, ghT, gdh, gdw): g.free()
    return f"linear-CE fwd+bwd M={M} V={V}  {ms:.3f} ms  {3*2.0*M*V*H/(ms/1e3)/1e12:5.0f} TFLOPS"


if __name__ == "__main__":
    print(f"ANCORA kernel perf — cuda-tile {ct.__version__}, sm_120a")
    print("=" * 64)
    for f in (bench_linear, bench_attn_fwd, bench_attn_bwd, bench_linear_ce):
        print("  " + f())
    print("-" * 64)
    print("  memory-bound (GB/s, % of 896 GB/s peak):")
    for f in (bench_rmsnorm, bench_rope):
        print("  " + f())
    print("=" * 64)
    print("  (kernel-only, CUDA-event timed, pre-uploaded data)")
