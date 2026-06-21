"""Full Qwen3-layer FORWARD MFU with the MXFP8 + fused-quant resident path, at the real
layer size (M=B*S). Companion to bench_layer_mfu.py (BF16 baseline, ~24% MFU). Measures the
END-TO-END chained forward (the honest "整层 MFU") plus a GEMM/attn/other split.

Fused: input_ln & post_ln → _rmsnorm_apply_q (emit fp8+scale, no bf16 round-trip) feed
q/k/v & gate/up; swiglu → _swiglu_fwd_q feeds down. Only o_proj keeps a standalone quant.
MXFP8 GEMM peak ~330, BF16 ~165 TFLOPS."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import Qwen3Config
from ancora.kernels.norm import _rmsnorm_stats, _rmsnorm_apply, _rmsnorm_apply_q, TM as NTM, TH, QB
from ancora.kernels.activation import _swiglu_fwd_q
from ancora.kernels.rope import _rope_fwd_tok, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_fwd_tok_q, BQ
from ancora.kernels.fused import (_residual_add, _gateup_swiglu_q, RTM, RTN, TT)
from ancora.kernels.linear import _fwd_mxfp8_bf16, _fwd_mxfp8_bf16_res, mxfp8_tile
from ancora.kernels.quant import _quant_mxfp8, QTM

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
# REAL measured achievable peaks on RTX 5080 Laptop (tests/hardware/_probe_peak*.py), NOT the
# old 165/330 assumption: BF16 GEMM caps at 80 (consumer BF16+FP32-acc throttle), MXFP8 real
# GEMM at 184 (memory-bound; compute ceiling 323). MFU normalized against THESE.
PEAK_BF16, PEAK_MXFP8 = 80.0, 184.0

def _bf(x): return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
def Z(sh, d=np.uint16): return GA.z(sh, d)

def time_ms(launch, iters=50, warmup=12):
    for _ in range(warmup): launch()
    so.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, SI)
    for _ in range(iters): launch()
    cudart.cudaEventRecord(t1, SI); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / iters


def bench(B, S):
    cfg = Qwen3Config()
    H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
    M, qd, kd = B * S, Hq * Dh, Hkv * Dh
    rng = np.random.default_rng(0)
    # weights as MXFP8 (random bytes ok for timing) + scales
    def WQ(K, N): return (GA((rng.integers(0, 255, (K, N))).astype(np.uint8)), GA(np.full((K // QB, N), 127, np.uint8)), K, N)
    W = {"q": WQ(H, qd), "k": WQ(H, kd), "v": WQ(H, kd), "o": WQ(qd, H),
         "gate": WQ(H, I), "up": WQ(H, I), "down": WQ(I, H)}
    gx = GA(_bf(rng.standard_normal((M, H)).astype(np.float32)))
    wln = GA(_bf(np.ones((1, H), np.float32))); wpln = GA(_bf(np.ones((1, H), np.float32)))
    wqn = GA(_bf(np.ones((1, Dh), np.float32))); wkn = GA(_bf(np.ones((1, Dh), np.float32)))
    cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta); gcos = GA(cosv); gsin = GA(sinv)
    gq = Z((M, qd)); gk = Z((M, kd)); gv = Z((M, kd)); gqn = Z((M, qd)); gkn = Z((M, kd))
    gqr = Z((M, qd)); gkr = Z((M, kd))   # token-major RoPE'd q/k (no head-major transpose)
    gL = Z((M * Hq, 1), np.float32)
    gotok = Z((M, qd)); gx2 = Z((M, H)); gg = Z((M, I)); gu = Z((M, I)); gmlp = Z((M, H)); gout = Z((M, H))
    r1 = Z((M, 1), np.float32); rq = Z((M * Hq, 1), np.float32); rk = Z((M * Hkv, 1), np.float32); r2 = Z((M, 1), np.float32)
    qh_f = Z((M, H), np.uint8); qh_s = Z((M, H // QB), np.uint8)
    qo_f = Z((M, qd), np.uint8); qo_s = Z((M, qd // QB), np.uint8)
    q2_f = Z((M, H), np.uint8); q2_s = Z((M, H // QB), np.uint8)
    qa_f = Z((M, I), np.uint8); qa_s = Z((M, I // QB), np.uint8)
    def Vw(g, sh): v = type("V", (), {})(); v.__cuda_array_interface__ = {"shape": sh, "typestr": "<u2", "data": (int(g.p), False), "version": 3}; return v
    NSB, NQB = S // TT, S // BQ; scale = 1.0 / math.sqrt(Dh)
    CUR = [SI]   # switchable launch stream (SI for timing, graph-builder stream for capture)

    def mxg(a_fp8, a_sc, nm, C):
        wfp8, wsc, K, N = W[nm]; TM, TN, TK = mxfp8_tile(N, K)
        ct.launch(CUR[0], (M // TM, N // TN, 1), _fwd_mxfp8_bf16, (a_fp8, wfp8, a_sc, wsc, C, K // TK, TM, TN, TK))
    def mxg_res(a_fp8, a_sc, nm, res, C):   # fused: out = GEMM + residual (megakernel fusion #2)
        wfp8, wsc, K, N = W[nm]; TM, TN, TK = mxfp8_tile(N, K)
        ct.launch(CUR[0], (M // TM, N // TN, 1), _fwd_mxfp8_bf16_res, (a_fp8, wfp8, a_sc, wsc, res, C, K // TK, TM, TN, TK))
    def rms(xb, wb, rstd, yb, rows, hh):
        ct.launch(CUR[0], (rows // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, eps))
        ct.launch(CUR[0], (rows // NTM, 1, 1), _rmsnorm_apply, (xb, wb, rstd, yb, hh // TH))
    def rms_q(xb, wb, rstd, fp8, sc, hh):
        ct.launch(CUR[0], (M // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, eps))
        ct.launch(CUR[0], (M // NTM, 1, 1), _rmsnorm_apply_q, (xb, wb, rstd, fp8, sc, hh // QB))

    def gemms():   # isolated: assumes fp8 inputs ready (pre-quantized once)
        mxg(qh_f, qh_s, "q", gq); mxg(qh_f, qh_s, "k", gk); mxg(qh_f, qh_s, "v", gv)
        mxg(qo_f, qo_s, "o", gx2); mxg(q2_f, q2_s, "gate", gg); mxg(q2_f, q2_s, "up", gu); mxg(qa_f, qa_s, "down", gmlp)
    def attn():   # token-major attention + FUSED output MXFP8 quant (writes qo_f/qo_s directly)
        ct.launch(CUR[0], (NQB, B * Hq, 1), _attn_fwd_tok_q, (gqr, gkr, gv, qo_f, qo_s, gL, NQB, NQB, Hq, Hkv, scale))
    def forward():   # full chained MXFP8 + fused-quant forward
        rms_q(gx, wln, r1, qh_f, qh_s, H)
        mxg(qh_f, qh_s, "q", gq); mxg(qh_f, qh_s, "k", gk); mxg(qh_f, qh_s, "v", gv)
        rms(Vw(gq, (M * Hq, Dh)), wqn, rq, Vw(gqn, (M * Hq, Dh)), M * Hq, Dh)
        rms(Vw(gk, (M * Hkv, Dh)), wkn, rk, Vw(gkn, (M * Hkv, Dh)), M * Hkv, Dh)
        ct.launch(CUR[0], (M // RRTM, Hq, 1), _rope_fwd_tok, (gqn, gcos, gsin, gqr, S // RRTM, Dh // 2))   # token-major RoPE (no transpose)
        ct.launch(CUR[0], (M // RRTM, Hkv, 1), _rope_fwd_tok, (gkn, gcos, gsin, gkr, S // RRTM, Dh // 2))
        attn()                                         # attention writes qo_f/qo_s (fp8) directly — no quant kernel
        mxg_res(qo_f, qo_s, "o", gx, gx2)              # fusion #2: o_proj + residual → gx2
        rms_q(gx2, wpln, r2, q2_f, q2_s, H)
        # MEGAKERNEL fusion #1: gate/up GEMM + SwiGLU + quant in one launch (gate/up never hit HBM)
        wgf, wgs, _, _ = W["gate"]; wuf, wus, _, _ = W["up"]
        ct.launch(CUR[0], (M // 128, I // 32, 1), _gateup_swiglu_q, (q2_f, q2_s, wgf, wgs, wuf, wus, qa_f, qa_s, H // 128, 128, 128))
        mxg_res(qa_f, qa_s, "down", gx2, gout)         # fusion #2: down_proj + residual → gout

    t_tot = time_ms(forward); t_g = time_ms(gemms); t_a = time_ms(attn)
    t_other = t_tot - t_g - t_a

    # CUDA-graph capture of the full forward: does eliminating per-kernel host launch
    # overhead shrink end-to-end time? forward() reads CUR[0] as its launch stream.
    t_graph = None
    try:
        for _ in range(3): forward()
        so.sync()
        gb = dev.create_graph_builder(); gb.begin_building()
        CUR[0] = int(gb.__cuda_stream__()[1])
        forward()
        CUR[0] = SI
        gb.end_building(); graph = gb.complete()
        for _ in range(5): graph.launch(so)
        so.sync(); _, e0 = cudart.cudaEventCreate(); _, e1 = cudart.cudaEventCreate()
        cudart.cudaEventRecord(e0, SI)
        for _ in range(50): graph.launch(so)
        cudart.cudaEventRecord(e1, SI); cudart.cudaEventSynchronize(e1)
        t_graph = cudart.cudaEventElapsedTime(e0, e1)[1] / 50
    except Exception as ex:
        t_graph = f"FAIL {type(ex).__name__}: {str(ex)[:80]}"
    gemm_fl = 2.0 * M * (H * qd + 2 * H * kd + qd * H + 3 * H * I)
    attn_fl = 2.0 * B * Hq * S * S * Dh
    flops = gemm_fl + attn_fl
    eff = flops / (t_tot / 1e3) / 1e12
    # ACHIEVABLE-ceiling MFU: ideal time if GEMMs ran at MXFP8 peak + attn at BF16 peak.
    ideal_ms = (gemm_fl / PEAK_MXFP8 + attn_fl / PEAK_BF16) / 1e12 * 1e3
    mfu = ideal_ms / t_tot * 100
    print(f"B={B} S={S} M={M}:")
    print(f"  GEMMs (isolated)  {t_g:.3f} ms  {gemm_fl/(t_g/1e3)/1e12:5.0f} TFLOPS  ({gemm_fl/(t_g/1e3)/1e12/PEAK_MXFP8*100:.0f}% of real MXFP8 peak {PEAK_MXFP8:.0f})")
    print(f"  Attn  (isolated)  {t_a:.3f} ms  {attn_fl/(t_a/1e3)/1e12:5.0f} TFLOPS  ({attn_fl/(t_a/1e3)/1e12/PEAK_BF16*100:.0f}% of real BF16 peak {PEAK_BF16:.0f})")
    print(f"  other (norm+rope+swiglu+quant+resid+transpose) {t_other:.3f} ms  ({t_other/t_tot*100:.0f}% of total)")
    print(f"  TOTAL end-to-end  {t_tot:.3f} ms  → {eff:5.0f} TFLOPS effective")
    print(f"    → MFU vs ACHIEVABLE ceiling (MXFP8 GEMM {PEAK_MXFP8:.0f} + BF16 attn {PEAK_BF16:.0f}): {mfu:.0f}%")
    print(f"  time split: GEMM {t_g/t_tot*100:.0f}% | attn {t_a/t_tot*100:.0f}% | other {t_other/t_tot*100:.0f}%")
    if isinstance(t_graph, float):
        ge = flops / (t_graph / 1e3) / 1e12
        print(f"  CUDA-graph replay {t_graph:.3f} ms  → {ge:5.0f} TFLOPS = {ge/PEAK_BF16*100:.0f}% bf16 peak "
              f"({t_tot/t_graph:.2f}x vs per-kernel launch — host-overhead share)")
    else:
        print(f"  CUDA-graph: {t_graph}")


if __name__ == "__main__":
    print("Qwen3 layer FORWARD MFU — MXFP8 + fused-quant resident path"); print("=" * 70)
    bench(1, 2048); print("-" * 70)
    bench(4, 2048)
    print("=" * 70)
    print("  baseline (BF16, bench_layer_mfu.py): ~38-40 TFLOPS, 23-24% MFU vs bf16 peak")
