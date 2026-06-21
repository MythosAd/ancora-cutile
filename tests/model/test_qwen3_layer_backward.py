"""Qwen3 decoder layer BACKWARD — kernel grads vs an fp64 analytical reference, and
the reference itself validated by finite differences (so a shared math error can't
hide). Checks d_x and every weight grad. Keep — re-run after kernel/toolkit changes."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.kernels.rope import build_cos_sin

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


# ── fp64 reference forward (returns out + cache) and backward (analytical VJP) ──

def _rope_pair(S, Dh, base):
    c, s = build_cos_sin(S, Dh, base)
    return (np.concatenate([c, c], -1)[None, None].astype(np.float64),
            np.concatenate([s, s], -1)[None, None].astype(np.float64))


def ref_fwd(xt, w, cfg):
    B, S = cfg._B, cfg._S; H = cfg.hidden
    Hq, Hkv, Dh, I, G = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.n_heads // cfg.n_kv_heads
    M = B * S; eps = cfg.eps
    W = {k: v.astype(np.float64) for k, v in w.items()}

    def rms(z, gn):
        r = 1.0 / np.sqrt((z * z).mean(-1, keepdims=True) + eps)
        return z * r * gn
    cosf, sinf = _rope_pair(S, Dh, cfg.rope_theta); d = Dh // 2
    def rope(x): return x * cosf + np.concatenate([-x[..., d:], x[..., :d]], -1) * sinf

    h1 = rms(xt, W["input_ln"])
    q = h1 @ W["q_proj"]; k = h1 @ W["k_proj"]; v = h1 @ W["v_proj"]
    qn = rms(q.reshape(M * Hq, Dh), W["q_norm"]); kn = rms(k.reshape(M * Hkv, Dh), W["k_norm"])
    qh = qn.reshape(B, S, Hq, Dh).transpose(0, 2, 1, 3); kh = kn.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    vh = v.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    qr = rope(qh); kr = rope(kh)
    k_e = np.repeat(kr, G, 1); v_e = np.repeat(vh, G, 1)
    sc = np.einsum('bhqd,bhkd->bhqk', qr, k_e) / math.sqrt(Dh)
    sc = np.where(np.triu(np.ones((S, S), bool), 1)[None, None], -np.inf, sc)
    P = np.exp(sc - sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    o = np.einsum('bhqk,bhkd->bhqd', P, v_e)
    o_tok = o.transpose(0, 2, 1, 3).reshape(M, Hq * Dh)
    x2 = xt + o_tok @ W["o_proj"]
    h2 = rms(x2, W["post_ln"])
    gg = h2 @ W["gate_proj"]; uu = h2 @ W["up_proj"]
    a = (gg / (1.0 + np.exp(-gg))) * uu
    out = x2 + a @ W["down_proj"]
    cache = dict(xt=xt, h1=h1, q=q, k=k, qr=qr, kr=kr, vh=vh, v_e=v_e, k_e=k_e, P=P, o=o,
                 o_tok=o_tok, x2=x2, h2=h2, gg=gg, uu=uu, a=a, W=W, cosf=cosf, sinf=sinf)
    return out, cache


def ref_bwd(d_out, c, cfg):
    B, S = cfg._B, cfg._S; H = cfg.hidden
    Hq, Hkv, Dh, I, G = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.n_heads // cfg.n_kv_heads
    M = B * S; eps = cfg.eps; W = c["W"]; half = Dh // 2

    def rms_bwd(z, gn, dy):
        D = z.shape[-1]
        r = 1.0 / np.sqrt((z * z).mean(-1, keepdims=True) + eps)
        cc_ = (dy * gn * z).sum(-1, keepdims=True)
        dz = r * dy * gn - r**3 * z * cc_ / D
        dgn = (dy * z * r).sum(0)
        return dz, dgn
    def rope_bwd(dy):
        return dy * c["cosf"] + np.concatenate([dy[..., half:], -dy[..., :half]], -1) * c["sinf"]

    g = {}
    d = d_out
    # MLP
    d_x2 = d.copy()
    g["down_proj"] = c["a"].T @ d
    d_a = d @ W["down_proj"].T
    sig = 1.0 / (1.0 + np.exp(-c["gg"])); silu = c["gg"] * sig
    d_gg = d_a * c["uu"] * (sig * (1.0 + c["gg"] * (1.0 - sig)))
    d_uu = d_a * silu
    g["gate_proj"] = c["h2"].T @ d_gg; g["up_proj"] = c["h2"].T @ d_uu
    d_h2 = d_gg @ W["gate_proj"].T + d_uu @ W["up_proj"].T
    d_x2p, g["post_ln"] = rms_bwd(c["x2"], W["post_ln"], d_h2)
    d_x2 = d_x2 + d_x2p
    # Attention
    d_attn = d_x2
    g["o_proj"] = c["o_tok"].T @ d_attn
    d_o_tok = d_attn @ W["o_proj"].T
    d_o = d_o_tok.reshape(B, S, Hq, Dh).transpose(0, 2, 1, 3)
    d_v_e = np.einsum('bhqk,bhqd->bhkd', c["P"], d_o)
    d_P = np.einsum('bhqd,bhkd->bhqk', d_o, c["v_e"])
    d_S = c["P"] * (d_P - (d_P * c["P"]).sum(-1, keepdims=True)) / math.sqrt(Dh)
    d_qr = np.einsum('bhqk,bhkd->bhqd', d_S, c["k_e"])
    d_k_e = np.einsum('bhqk,bhqd->bhkd', d_S, c["qr"])
    d_kr = d_k_e.reshape(B, Hkv, G, S, Dh).sum(2); d_vh = d_v_e.reshape(B, Hkv, G, S, Dh).sum(2)
    d_qn = rope_bwd(d_qr).transpose(0, 2, 1, 3).reshape(M * Hq, Dh)
    d_kn = rope_bwd(d_kr).transpose(0, 2, 1, 3).reshape(M * Hkv, Dh)
    d_v = d_vh.transpose(0, 2, 1, 3).reshape(M, Hkv * Dh)
    d_q, g["q_norm"] = rms_bwd(c["q"].reshape(M * Hq, Dh), W["q_norm"], d_qn)
    d_k, g["k_norm"] = rms_bwd(c["k"].reshape(M * Hkv, Dh), W["k_norm"], d_kn)
    d_q = d_q.reshape(M, Hq * Dh); d_k = d_k.reshape(M, Hkv * Dh)
    g["q_proj"] = c["h1"].T @ d_q; g["k_proj"] = c["h1"].T @ d_k; g["v_proj"] = c["h1"].T @ d_v
    d_h1 = d_q @ W["q_proj"].T + d_k @ W["k_proj"].T + d_v @ W["v_proj"].T
    d_xp, g["input_ln"] = rms_bwd(c["xt"], W["input_ln"], d_h1)
    d_x = d_x2 + d_xp
    return d_x, g


def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def test_reference_with_fd(cfg, w):
    """Validate the fp64 reference backward via central finite differences on d_x and
    a weight (input_ln). Pure fp64 → FD is reliable."""
    print("--- validate fp64 reference backward via finite differences ---")
    M, H = cfg._B * cfg._S, cfg.hidden
    rng = np.random.default_rng(3)
    xt = (rng.standard_normal((M, H)) * 0.5)
    cot = rng.standard_normal((M, H))
    out, cache = ref_fwd(xt, w, cfg)
    dx_ref, g_ref = ref_bwd(cot, cache, cfg)

    def loss(xt_): return float((ref_fwd(xt_, w, cfg)[0] * cot).sum())
    eps = 1e-5; ok = True
    errs = []
    for _ in range(8):
        i, j = rng.integers(M), rng.integers(H)
        xp = xt.copy(); xp[i, j] += eps; xm = xt.copy(); xm[i, j] -= eps
        fd = (loss(xp) - loss(xm)) / (2 * eps)
        errs.append(abs(fd - dx_ref[i, j]) / (abs(fd) + 1e-6))
    e = max(errs); ok &= e < 1e-4
    print(f"  d_x   vs FD: max rel {e:.2e}  {'OK' if e < 1e-4 else 'FAIL'}")

    # one weight: input_ln (D,)
    w2 = {k: v.copy() for k, v in w.items()}
    def lossw(vec):
        w2["input_ln"] = vec
        return float((ref_fwd(xt, w2, cfg)[0] * cot).sum())
    base = w["input_ln"].astype(np.float64); errs = []
    for _ in range(6):
        j = rng.integers(H); vp = base.copy(); vp[j] += eps; vm = base.copy(); vm[j] -= eps
        fd = (lossw(vp) - lossw(vm)) / (2 * eps)
        errs.append(abs(fd - g_ref["input_ln"][j]) / (abs(fd) + 1e-6))
    ew = max(errs); ok &= ew < 1e-4
    print(f"  d_input_ln vs FD: max rel {ew:.2e}  {'OK' if ew < 1e-4 else 'FAIL'}")
    return ok


def test_kernel_vs_reference():
    print("--- kernel backward vs fp64 reference ---")
    cfg = Qwen3Config(); cfg._B, cfg._S = 1, 128
    layer = TransformerLayer(cfg, seed=0)
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((cfg._B, cfg._S, cfg.hidden)) * 0.5).astype(np.float32)
    cot = (rng.standard_normal((cfg._B, cfg._S, cfg.hidden)) * 1.0).astype(np.float32)

    ref_ok = test_reference_with_fd(cfg, layer.w)

    _, cache_k = layer.forward(x, si, return_cache=True)
    dx_k, g_k = layer.backward(cot, cache_k, si)
    _, ref_cache = ref_fwd(x.reshape(-1, cfg.hidden).astype(np.float64), layer.w, cfg)
    dx_r, g_r = ref_bwd(cot.reshape(-1, cfg.hidden).astype(np.float64), ref_cache, cfg)

    ok = ref_ok
    e = rel(dx_k.reshape(-1, cfg.hidden), dx_r); o = e < 0.05; ok &= o
    print(f"  d_x: {e*100:.2f}%  {'OK' if o else 'FAIL'}")
    for name in ["input_ln", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
                 "o_proj", "post_ln", "gate_proj", "up_proj", "down_proj"]:
        e = rel(g_k[name], g_r[name]); o = e < 0.06; ok &= o
        print(f"  d_{name:10s}: {e*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cfg = Qwen3Config()
    print(f"Qwen3 layer BACKWARD — hidden={cfg.hidden} Hq={cfg.n_heads} Hkv={cfg.n_kv_heads} Dh={cfg.head_dim}")
    print("=" * 64)
    ok = test_kernel_vs_reference()
    print("=" * 64)
    print(f"  {'PASS' if ok else 'FAIL'}")
