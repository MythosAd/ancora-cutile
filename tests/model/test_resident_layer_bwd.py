"""Full device-resident layer BACKWARD (attention + MLP) vs host layer.backward. Capstone:
completes the device-resident fast training step (fwd 97× + bwd, all on-device)."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.norm import (_rmsnorm_bwd_dx, _rmsnorm_dw_part, _rmsnorm_dw_reduce,
                                  TM as NTM, TH, TD, PART, f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32)
from ancora.kernels.activation import _swiglu_bwd, TM as STM, TI
from ancora.kernels.rope import _rope_bwd, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_bwd_dq, _attn_bwd_dkdv, BQ, D as DH
from ancora.kernels.fused import (_gemm_dx, _gemm_dW, _residual_add, _tok_to_head, _head_to_tok,
                                  _head_to_tok_f32, _attn_delta, _cast64, RTM, RTN, TT, DTM)

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])
cfg = Qwen3Config(); B, S = 1, 128
H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
M, qd, kd, G = B * S, Hq * Dh, Hkv * Dh, cfg.n_heads // cfg.n_kv_heads
T = 64; scale = 1.0 / math.sqrt(Dh); NSB, NQB = S // TT, S // BQ
layer = TransformerLayer(cfg, seed=0); rng = np.random.default_rng(1)
x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
dout = _bf((rng.standard_normal((B, S, H)) * 0.1).astype(np.float32))

# ── host reference (validated path) ──
out, cache = layer.forward(x, si, return_cache=True)
d_x_ref, gref = layer.backward(dout, cache, si)

# ── device buffers ──
class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
def Z(sh, dt=np.uint16): return GA(np.zeros(sh, dt))
def Vw(g, shape):
    v = type("V", (), {})(); v.__cuda_array_interface__ = {"shape": shape, "typestr": np.dtype(g.dt).str, "data": (int(g.p), False), "version": 3}; return v
def Wb(name): return GA(f32bf(layer.w[name]))
def Cb(key, shape=None): return GA(f32bf(cache[key].reshape(shape) if shape else cache[key]))
def Cf(key, shape): return GA(cache[key].reshape(shape).astype(np.float32))

# cache → device
gdout = GA(f32bf(dout.reshape(M, H)))
gx = Cb("x"); gh1 = Cb("h1"); gq = Cb("q"); gk = Cb("k"); gqr = Cb("qr", (M * Hq, Dh)); gkr = Cb("kr", (M * Hkv, Dh))
gvh = Cb("vh", (M * Hkv, Dh)); gotok = Cb("o_tok"); gx2 = Cb("x2"); gh2 = Cb("h2"); ggate = Cb("gate"); gup = Cb("up"); gact = Cb("act")
gOatt = Cf("o_attn", (M * Hq, Dh)); gL = Cf("L", (M * Hq, 1))
gr1 = Cf("rstd1", (M, 1)); grq = Cf("rstd_q", (M * Hq, 1)); grk = Cf("rstd_k", (M * Hkv, 1)); gr2 = Cf("rstd2", (M, 1))
wiln = GA(f32bf(layer.w["input_ln"].reshape(1, H))); wpln = GA(f32bf(layer.w["post_ln"].reshape(1, H)))
wqn = GA(f32bf(layer.w["q_norm"].reshape(1, Dh))); wkn = GA(f32bf(layer.w["k_norm"].reshape(1, Dh)))
wq = Wb("q_proj"); wk = Wb("k_proj"); wv = Wb("v_proj"); wo = Wb("o_proj"); wg = Wb("gate_proj"); wu = Wb("up_proj"); wdn = Wb("down_proj")
cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta); gcos = GA(cosv); gsin = GA(sinv)

# intermediates / grads
gda = Z((M, I)); gdg = Z((M, I)); gdu = Z((M, I)); gdh2a = Z((M, H)); gdh2b = Z((M, H)); gdh2 = Z((M, H)); gdx2m = Z((M, H)); gdx2 = Z((M, H))
gdotok = Z((M, qd)); gdohm = Z((M * Hq, Dh)); gDelta = Z((M * Hq, 1), np.float32)
gdqr = Z((M * Hq, Dh), np.float32); gdkr = Z((M * Hkv, Dh), np.float32); gdvh = Z((M * Hkv, Dh), np.float32)
gdqrb = Z((M * Hq, Dh)); gdkrb = Z((M * Hkv, Dh)); gdqnhm = Z((M * Hq, Dh)); gdknhm = Z((M * Hkv, Dh))
gdqn = Z((M, qd)); gdkn = Z((M, kd)); gdv = Z((M, kd)); gdq = Z((M, qd)); gdk = Z((M, kd))
gdh1q = Z((M, H)); gdh1k = Z((M, H)); gdh1v = Z((M, H)); gdh1t = Z((M, H)); gdh1 = Z((M, H)); gdxa = Z((M, H)); gdx = Z((M, H))
# weight grads (f32)
Gdn = Z((I, H), np.float32); Gg = Z((H, I), np.float32); Gu = Z((H, I), np.float32); Gpl = Z((1, H), np.float32)
Go = Z((qd, H), np.float32); Gqn = Z((1, Dh), np.float32); Gkn = Z((1, Dh), np.float32)
Gq = Z((H, qd), np.float32); Gk = Z((H, kd), np.float32); Gv = Z((H, kd), np.float32); Gil = Z((1, H), np.float32)
part = Z((PART, max(H, Dh)), np.float32)


def dx(dy, W, out, K, N): ct.launch(si, (M // T, K // T, 1), _gemm_dx, (dy, W, out, N // T, T, T, T))
def dW(xb, dy, out, K, N): ct.launch(si, (K // T, N // T, 1), _gemm_dW, (xb, dy, out, M // T, T, T, T))
def radd(a, b, o): ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (a, b, o))
def rms_bwd(xb, wb, dy, rstd, dxo, gw, rows, hh, pbuf):
    ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_bwd_dx, (xb, wb, dy, rstd, dxo, hh // TH, 1.0 / hh))
    mb = rows // NTM; bpp = (mb + PART - 1) // PART
    ct.launch(si, (hh // TD, PART, 1), _rmsnorm_dw_part, (xb, dy, rstd, Vw(pbuf, (PART, hh)), mb, bpp))
    ct.launch(si, (hh // TD, 1, 1), _rmsnorm_dw_reduce, (Vw(pbuf, (PART, hh)), gw))


def bwd():
    # MLP
    dx(gdout, wdn, gda, I, H); dW(gact, gdout, Gdn, I, H)
    ct.launch(si, (M // STM, I // TI, 1), _swiglu_bwd, (ggate, gup, gda, gdg, gdu))
    dx(gdg, wg, gdh2a, H, I); dW(gh2, gdg, Gg, H, I)
    dx(gdu, wu, gdh2b, H, I); dW(gh2, gdu, Gu, H, I)
    radd(gdh2a, gdh2b, gdh2)
    rms_bwd(gx2, wpln, gdh2, gr2, gdx2m, Gpl, M, H, part)
    radd(gdout, gdx2m, gdx2)
    # Attention
    dx(gdx2, wo, gdotok, qd, H); dW(gotok, gdx2, Go, qd, H)
    ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (gdotok, gdohm, Hq, NSB))
    ct.launch(si, (M * Hq // DTM, 1, 1), _attn_delta, (gOatt, gdohm, gDelta))
    ct.launch(si, (NQB, B * Hq, 1), _attn_bwd_dq, (gqr, gkr, gvh, gdohm, gL, gDelta, gdqr, NQB, NQB, Hq, Hkv, scale))
    ct.launch(si, (NQB, B * Hkv, 1), _attn_bwd_dkdv, (gqr, gkr, gvh, gdohm, gL, gDelta, gdkr, gdvh, NQB, NQB, Hq, Hkv, G, scale))
    ct.launch(si, (M * Hq // DTM, 1, 1), _cast64, (gdqr, gdqrb)); ct.launch(si, (M * Hkv // DTM, 1, 1), _cast64, (gdkr, gdkrb))
    ct.launch(si, (S // RRTM, B * Hq, 1), _rope_bwd, (gdqrb, gcos, gsin, gdqnhm, S // RRTM, Dh // 2))
    ct.launch(si, (S // RRTM, B * Hkv, 1), _rope_bwd, (gdkrb, gcos, gsin, gdknhm, S // RRTM, Dh // 2))
    ct.launch(si, (B * Hq, NSB, 1), _head_to_tok, (gdqnhm, gdqn, Hq, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _head_to_tok, (gdknhm, gdkn, Hkv, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _head_to_tok_f32, (gdvh, gdv, Hkv, NSB))
    rms_bwd(Vw(gq, (M * Hq, Dh)), wqn, Vw(gdqn, (M * Hq, Dh)), grq, Vw(gdq, (M * Hq, Dh)), Gqn, M * Hq, Dh, part)
    rms_bwd(Vw(gk, (M * Hkv, Dh)), wkn, Vw(gdkn, (M * Hkv, Dh)), grk, Vw(gdk, (M * Hkv, Dh)), Gkn, M * Hkv, Dh, part)
    dx(gdq, wq, gdh1q, H, qd); dW(gh1, gdq, Gq, H, qd)
    dx(gdk, wk, gdh1k, H, kd); dW(gh1, gdk, Gk, H, kd)
    dx(gdv, wv, gdh1v, H, kd); dW(gh1, gdv, Gv, H, kd)
    radd(gdh1q, gdh1k, gdh1t); radd(gdh1t, gdh1v, gdh1)
    rms_bwd(gx, wiln, gdh1, gr1, gdxa, Gil, M, H, part)
    radd(gdx2, gdxa, gdx)


if __name__ == "__main__":
    print(f"Device-resident FULL layer backward  B={B} S={S}"); print("=" * 60)
    bwd(); cudart.cudaStreamSynchronize(si)
    rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    checks = [("d_x", bf32(gdx.np()).reshape(B, S, H), d_x_ref), ("down", Gdn.np(), gref["down_proj"]),
              ("gate", Gg.np(), gref["gate_proj"]), ("up", Gu.np(), gref["up_proj"]), ("post_ln", Gpl.np().reshape(H), gref["post_ln"]),
              ("o_proj", Go.np(), gref["o_proj"]), ("q_norm", Gqn.np().reshape(Dh), gref["q_norm"]), ("k_norm", Gkn.np().reshape(Dh), gref["k_norm"]),
              ("q_proj", Gq.np(), gref["q_proj"]), ("k_proj", Gk.np(), gref["k_proj"]), ("v_proj", Gv.np(), gref["v_proj"]), ("input_ln", Gil.np().reshape(H), gref["input_ln"])]
    ok = True
    for nm, a, b in checks:
        e = rel(a, b); o = e < 0.05; ok &= o
        print(f"  d_{nm:10s} {e*100:.2f}%  {'OK' if o else 'FAIL'}")
    print("=" * 60); print(f"  {'PASS — full device-resident training step (fwd+bwd) works' if ok else 'FAIL'}")
