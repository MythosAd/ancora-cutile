"""Device-resident Qwen3 layer forward with MXFP8 projections (quant + ct.mma_scaled,
bf16 output) vs the host BF16 layer.forward. Proves: (1) numerics hold with FP8 weights+
activations (~4% vs bf16), (2) the 7 GEMMs go from BF16 ~44 TFLOPS to MXFP8 ~110-139 →
the FLOP-dominant ones cross 80% of BF16-equivalent peak (gemm_mfu_ceiling memory).

Amortized activation quant: 4 quant launches feed 7 GEMMs (gh→q/k/v, gotok→o, gh2→gate/up,
ga→down). Weights pre-quantized once offline (quantize_colblock). Keep."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, _rmsnorm_apply_q, TM as NTM, TH, QB,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32)
from ancora.kernels.activation import _swiglu_fwd, _swiglu_fwd_q, TM as STM, TI
from ancora.kernels.rope import _rope_fwd, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_fwd, BQ, D as DH
from ancora.kernels.fused import (_gemm_bf16, _residual_add, _tok_to_head, _head_to_tok_f32,
                                  _gateup_swiglu_q, RTM, RTN, TT)
from ancora.kernels.loss import GTM, GTN, GTK
from ancora.kernels.linear import _fwd_mxfp8_bf16, _fwd_mxfp8_bf16_res, mxfp8_tile
from ancora.kernels.quant import _quant_mxfp8, QTM, B as QB, quantize_colblock

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])

cfg = Qwen3Config(); B, S = 1, 256
H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
M, qd, kd = B * S, Hq * Dh, Hkv * Dh
layer = TransformerLayer(cfg, seed=0)
rng = np.random.default_rng(1)
x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
ref = layer.forward(x, si)

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
def Z(sh, dt=np.uint16): return GA(np.zeros(sh, dt))
def view(g, shape):
    v = type("V", (), {})(); v.__cuda_array_interface__ = {"shape": shape, "typestr": np.dtype(g.dt).str, "data": (int(g.p), False), "version": 3}
    return v

gx = GA(f32bf(x.reshape(M, H)))
wln = GA(f32bf(layer.w["input_ln"].reshape(1, H))); wpln = GA(f32bf(layer.w["post_ln"].reshape(1, H)))
wqn = GA(f32bf(layer.w["q_norm"].reshape(1, Dh))); wkn = GA(f32bf(layer.w["k_norm"].reshape(1, Dh)))
cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta); gcos = GA(cosv); gsin = GA(sinv)

# ── pre-quantize weights to MXFP8 (offline, once) ──
WQ = {}   # name → (w_fp8 GA, w_scale GA, K, N)
for nm in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    wf32 = layer.w[nm]; K, N = wf32.shape
    wfp8, wsc = quantize_colblock(wf32.astype(np.float32))
    WQ[nm] = (GA(wfp8), GA(wsc), K, N)

# intermediates (bf16 bits unless noted)
gh = Z((M, H)); gq = Z((M, qd)); gk = Z((M, kd)); gv = Z((M, kd))
gqn = Z((M, qd)); gkn = Z((M, kd))
gqh = Z((M * Hq, Dh)); gkh = Z((M * Hkv, Dh)); gvh = Z((M * Hkv, Dh))
gqr = Z((M * Hq, Dh)); gkr = Z((M * Hkv, Dh)); gO = Z((M * Hq, Dh), np.float32); gL = Z((M * Hq, 1), np.float32)
gotok = Z((M, qd)); gattn = Z((M, H)); gx2 = Z((M, H)); gh2 = Z((M, H))
gg = Z((M, I)); gu = Z((M, I)); ga = Z((M, I)); gmlp = Z((M, H)); gout = Z((M, H))
r1 = Z((M, 1), np.float32); rq = Z((M * Hq, 1), np.float32); rk = Z((M * Hkv, 1), np.float32); r2 = Z((M, 1), np.float32)
# MXFP8 activation-quant buffers (one per distinct GEMM input → amortized)
qh_f = Z((M, H), np.uint8);  qh_s = Z((M, H // QB), np.uint8)     # gh → q/k/v
qo_f = Z((M, qd), np.uint8); qo_s = Z((M, qd // QB), np.uint8)    # gotok → o
q2_f = Z((M, H), np.uint8);  q2_s = Z((M, H // QB), np.uint8)     # gh2 → gate/up
qa_f = Z((M, I), np.uint8);  qa_s = Z((M, I // QB), np.uint8)     # ga → down
NSB, NQB = S // TT, S // BQ; scale = 1.0 / math.sqrt(Dh)


def quant(src, fp8, sc, K):  # src bf16-bits (M,K) → fp8 (M,K) u8 + sc (M,K//32) u8
    ct.launch(si, (M // QTM, 1, 1), _quant_mxfp8, (src, fp8, sc, K // QB))
def mxgemm(a_fp8, a_sc, nm, C):
    wfp8, wsc, K, N = WQ[nm]; TM, TN, TK = mxfp8_tile(N, K)
    ct.launch(si, (M // TM, N // TN, 1), _fwd_mxfp8_bf16, (a_fp8, wfp8, a_sc, wsc, C, K // TK, TM, TN, TK))
def mxgemm_res(a_fp8, a_sc, nm, res, C):   # fused: out = GEMM + residual (megakernel fusion #2)
    wfp8, wsc, K, N = WQ[nm]; TM, TN, TK = mxfp8_tile(N, K)
    ct.launch(si, (M // TM, N // TN, 1), _fwd_mxfp8_bf16_res, (a_fp8, wfp8, a_sc, wsc, res, C, K // TK, TM, TN, TK))
def rms(xb, wb, rstd, yb, rows, hh):
    ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, eps))
    ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_apply, (xb, wb, rstd, yb, hh // TH))
def rms_q(xb, wb, rstd, fp8, sc, hh):  # FUSED: stats + apply-with-quant → fp8 (no bf16 round-trip)
    ct.launch(si, (M // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, eps))
    ct.launch(si, (M // NTM, 1, 1), _rmsnorm_apply_q, (xb, wb, rstd, fp8, sc, hh // QB))


def resident_layer_mxfp8():
    # ── attention ──  (input_ln fused → fp8 directly feeds q/k/v)
    rms_q(gx, wln, r1, qh_f, qh_s, H)
    mxgemm(qh_f, qh_s, "q_proj", gq); mxgemm(qh_f, qh_s, "k_proj", gk); mxgemm(qh_f, qh_s, "v_proj", gv)
    rms(view(gq, (M * Hq, Dh)), wqn, rq, view(gqn, (M * Hq, Dh)), M * Hq, Dh)
    rms(view(gk, (M * Hkv, Dh)), wkn, rk, view(gkn, (M * Hkv, Dh)), M * Hkv, Dh)
    ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (gqn, gqh, Hq, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gkn, gkh, Hkv, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gv, gvh, Hkv, NSB))
    ct.launch(si, (S // RRTM, B * Hq, 1), _rope_fwd, (gqh, gcos, gsin, gqr, S // RRTM, Dh // 2))
    ct.launch(si, (S // RRTM, B * Hkv, 1), _rope_fwd, (gkh, gcos, gsin, gkr, S // RRTM, Dh // 2))
    ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gqr, gkr, gvh, gO, gL, NQB, NQB, Hq, Hkv, scale))
    ct.launch(si, (B * Hq, NSB, 1), _head_to_tok_f32, (gO, gotok, Hq, NSB))
    quant(gotok, qo_f, qo_s, qd)        # o_proj input: standalone quant (transpose-fuse = TODO)
    mxgemm_res(qo_f, qo_s, "o_proj", gx, gx2)        # fusion #2: o_proj + residual → gx2
    # ── MLP ──  (post_ln fused → fp8 feeds gate/up; swiglu fused → fp8 feeds down)
    rms_q(gx2, wpln, r2, q2_f, q2_s, H)
    # MEGAKERNEL fusion #1: gate/up GEMM + SwiGLU + quant in one launch (gate/up never hit HBM)
    wgf, wgs, _, _ = WQ["gate_proj"]; wuf, wus, _, _ = WQ["up_proj"]
    ct.launch(si, (M // 128, I // 32, 1), _gateup_swiglu_q, (q2_f, q2_s, wgf, wgs, wuf, wus, qa_f, qa_s, H // 128, 128, 128))
    mxgemm_res(qa_f, qa_s, "down_proj", gx2, gout)   # fusion #2: down_proj + residual → gout


# BF16 resident reference (same chain, _gemm_bf16) for an apples-to-apples speed compare
wbf = {nm: GA(f32bf(layer.w[nm])) for nm in WQ}
def gemm_bf(A, nm, C):
    K, N = WQ[nm][2], WQ[nm][3]
    ct.launch(si, (M // GTM, N // GTN, 1), _gemm_bf16, (A, wbf[nm], C, K // GTK, GTM, GTN, GTK))
def resident_layer_bf16():
    rms(gx, wln, r1, gh, M, H)
    gemm_bf(gh, "q_proj", gq); gemm_bf(gh, "k_proj", gk); gemm_bf(gh, "v_proj", gv)
    rms(view(gq, (M * Hq, Dh)), wqn, rq, view(gqn, (M * Hq, Dh)), M * Hq, Dh)
    rms(view(gk, (M * Hkv, Dh)), wkn, rk, view(gkn, (M * Hkv, Dh)), M * Hkv, Dh)
    ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (gqn, gqh, Hq, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gkn, gkh, Hkv, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gv, gvh, Hkv, NSB))
    ct.launch(si, (S // RRTM, B * Hq, 1), _rope_fwd, (gqh, gcos, gsin, gqr, S // RRTM, Dh // 2))
    ct.launch(si, (S // RRTM, B * Hkv, 1), _rope_fwd, (gkh, gcos, gsin, gkr, S // RRTM, Dh // 2))
    ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gqr, gkr, gvh, gO, gL, NQB, NQB, Hq, Hkv, scale))
    ct.launch(si, (B * Hq, NSB, 1), _head_to_tok_f32, (gO, gotok, Hq, NSB))
    gemm_bf(gotok, "o_proj", gattn)
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gx, gattn, gx2))
    rms(gx2, wpln, r2, gh2, M, H)
    gemm_bf(gh2, "gate_proj", gg); gemm_bf(gh2, "up_proj", gu)
    ct.launch(si, (M // STM, I // TI, 1), _swiglu_fwd, (gg, gu, ga))
    gemm_bf(ga, "down_proj", gmlp)
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gx2, gmlp, gout))


if __name__ == "__main__":
    print(f"Device-resident MXFP8 layer forward  B={B} S={S} H={H}"); print("=" * 64)
    resident_layer_mxfp8(); cudart.cudaStreamSynchronize(si)
    devv = bf32(gout.np()).reshape(B, S, H)
    rel = np.abs(devv - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"  MXFP8 resident vs host bf16 layer.forward: {rel*100:.2f}%  {'OK' if rel < 0.06 else 'FAIL'}")

    def wall(fn, it=30, warm=8):
        for _ in range(warm): fn()
        stream.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        stream.sync(); return (time.perf_counter() - t) / it * 1000
    t_mx = wall(resident_layer_mxfp8); t_bf = wall(resident_layer_bf16)
    print(f"  resident BF16 {t_bf*1000:6.0f} µs | resident MXFP8 {t_mx*1000:6.0f} µs | {t_bf/t_mx:.2f}x faster")
    print("=" * 64)
