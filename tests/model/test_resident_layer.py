"""Full device-resident Qwen3 layer (attention + MLP, all kernels chained on persistent
buffers) vs the host layer.forward. Validates correctness (bf16) + end-to-end speedup.
Completes the device-resident forward (the ~100× path)."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, TM as NTM, TH,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32)
from ancora.kernels.activation import _swiglu_fwd, TM as STM, TI
from ancora.kernels.loss import GTM, GTN, GTK
from ancora.kernels.rope import _rope_fwd, RTM as RRTM, build_cos_sin
from ancora.kernels.attention import _attn_fwd, BQ, D as DH
from ancora.kernels.fused import (_gemm_bf16, _residual_add, _tok_to_head, _head_to_tok_f32,
                                  RTM, RTN, TT)

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])

cfg = Qwen3Config(); B, S = 1, 128
H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
M, qd, kd = B * S, Hq * Dh, Hkv * Dh
layer = TransformerLayer(cfg, seed=0)
rng = np.random.default_rng(1)
x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))

# ── host reference ──
ref = layer.forward(x, si)

# ── persistent device buffers ──
class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
def Z(sh, dt=np.uint16): return GA(np.zeros(sh, dt))
def W(name): return GA(f32bf(layer.w[name]))
def view(g, shape):
    v = type("V", (), {})(); v.__cuda_array_interface__ = {"shape": shape, "typestr": np.dtype(g.dt).str, "data": (int(g.p), False), "version": 3}
    return v

gx = GA(f32bf(x.reshape(M, H)))
w = {n: W(n) for n in layer.w}
wln = GA(f32bf(layer.w["input_ln"].reshape(1, H))); wpln = GA(f32bf(layer.w["post_ln"].reshape(1, H)))
wqn = GA(f32bf(layer.w["q_norm"].reshape(1, Dh))); wkn = GA(f32bf(layer.w["k_norm"].reshape(1, Dh)))
cosv, sinv = build_cos_sin(S, Dh, cfg.rope_theta); gcos = GA(cosv); gsin = GA(sinv)
# intermediates
gh = Z((M, H)); gq = Z((M, qd)); gk = Z((M, kd)); gv = Z((M, kd))
gqn = Z((M, qd)); gkn = Z((M, kd))   # token-major (normed q,k)
# head-major buffers MUST be declared (B*H*S, Dh) — the declared shape sets ct.load/store strides
gqh = Z((M * Hq, Dh)); gkh = Z((M * Hkv, Dh)); gvh = Z((M * Hkv, Dh))
gqr = Z((M * Hq, Dh)); gkr = Z((M * Hkv, Dh)); gO = Z((M * Hq, Dh), np.float32); gL = Z((M * Hq, 1), np.float32)
gotok = Z((M, qd)); gattn = Z((M, H)); gx2 = Z((M, H)); gh2 = Z((M, H))
gg = Z((M, I)); gu = Z((M, I)); ga = Z((M, I)); gmlp = Z((M, H)); gout = Z((M, H))
r1 = Z((M, 1), np.float32); rq = Z((M * Hq, 1), np.float32); rk = Z((M * Hkv, 1), np.float32); r2 = Z((M, 1), np.float32)
NSB, NQB = S // TT, S // BQ; scale = 1.0 / math.sqrt(Dh)


def gemm(A, Bw, C, K, N): ct.launch(si, (M // GTM, N // GTN, 1), _gemm_bf16, (A, Bw, C, K // GTK, GTM, GTN, GTK))
def rms(xb, wb, rstd, yb, rows, hh):
    ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_stats, (xb, rstd, hh // TH, 1.0 / hh, eps))
    ct.launch(si, (rows // NTM, 1, 1), _rmsnorm_apply, (xb, wb, rstd, yb, hh // TH))


def resident_layer():
    # ── attention ──
    rms(gx, wln, r1, gh, M, H)
    gemm(gh, w["q_proj"], gq, H, qd); gemm(gh, w["k_proj"], gk, H, kd); gemm(gh, w["v_proj"], gv, H, kd)
    rms(view(gq, (M * Hq, Dh)), wqn, rq, view(gqn, (M * Hq, Dh)), M * Hq, Dh)
    rms(view(gk, (M * Hkv, Dh)), wkn, rk, view(gkn, (M * Hkv, Dh)), M * Hkv, Dh)
    ct.launch(si, (B * Hq, NSB, 1), _tok_to_head, (gqn, gqh, Hq, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gkn, gkh, Hkv, NSB))
    ct.launch(si, (B * Hkv, NSB, 1), _tok_to_head, (gv, gvh, Hkv, NSB))
    ct.launch(si, (S // RRTM, B * Hq, 1), _rope_fwd, (gqh, gcos, gsin, gqr, S // RRTM, Dh // 2))
    ct.launch(si, (S // RRTM, B * Hkv, 1), _rope_fwd, (gkh, gcos, gsin, gkr, S // RRTM, Dh // 2))
    ct.launch(si, (NQB, B * Hq, 1), _attn_fwd, (gqr, gkr, gvh, gO, gL, NQB, NQB, Hq, Hkv, scale))
    ct.launch(si, (B * Hq, NSB, 1), _head_to_tok_f32, (gO, gotok, Hq, NSB))
    gemm(gotok, w["o_proj"], gattn, qd, H)
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gx, gattn, gx2))
    # ── MLP ──
    rms(gx2, wpln, r2, gh2, M, H)
    gemm(gh2, w["gate_proj"], gg, H, I); gemm(gh2, w["up_proj"], gu, H, I)
    ct.launch(si, (M // STM, I // TI, 1), _swiglu_fwd, (gg, gu, ga))
    gemm(ga, w["down_proj"], gmlp, I, H)
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gx2, gmlp, gout))


if __name__ == "__main__":
    print(f"Device-resident FULL layer  B={B} S={S} H={H}")
    print("=" * 60)
    resident_layer(); cudart.cudaStreamSynchronize(si)
    devv = bf32(gout.np()).reshape(B, S, H)
    rel = np.abs(devv - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"  device-resident vs host layer.forward: {rel*100:.3f}%  {'OK' if rel < 0.02 else 'FAIL'}")

    def wall(fn, it=20, warm=5):
        for _ in range(warm): fn()
        stream.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        return (time.perf_counter() - t) / it * 1000
    th = wall(lambda: layer.forward(x, si)); td = wall(lambda: (resident_layer(), stream.sync()))
    print(f"  host-glue layer {th*1000:7.0f} µs | device-resident {td*1000:6.0f} µs | {th/td:.0f}x faster")
    print("=" * 60)
