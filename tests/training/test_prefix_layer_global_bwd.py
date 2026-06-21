"""Prefix-shared GLOBAL (NoPE) layer BACKWARD (rl/prefix_resident.prefix_layer_global_bwd) vs a naive
NoPE dense layer fwd+bwd on the G replicated [prompt, completion_i] (B=G). MoE global layers are NoPE
→ no rope-backward anywhere. Training-equivalence contract (same as the validated Qwen3 _prefix_layer_bwd):
  - weight grads == naive G-replicated layer grads
  - suffix input grad == naive d_in[:, Sp:]
  - prompt input grad == Σ_G naive d_in[:, :Sp]  (prompt processed once carries the summed grad)
Incoming prompt grad for the prefix path is G·d_pe (the sum of the per-copy incoming prompt grads);
top layer (GRPO, loss only on completions) → d_pe = 0.

Attention is the churn-free resident PrefixGlobalAttn; the per-token glue uses host helpers (alloc-
churn) → retry the case until the prefix forward is BITWISE-equal to the naive suffix, then trust grads."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import (Qwen3Config, TransformerLayer, linear_bf16,
                                      linear_bf16_backward, _bf)
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.activation import swiglu_forward, swiglu_backward
from ancora.kernels.attention import flash_attn_forward, flash_attn_backward, D
from ancora.rl.prefix_resident import PrefixGlobalAttn, prefix_layer_global, prefix_layer_global_bwd

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _naive_nope_fwd(w, full, cfg):
    """NoPE dense layer forward with cache, (G,S,H) → (G,S,H). TransformerLayer.forward minus RoPE."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    G, S, _ = full.shape; M = G * S; qd = Hq * Dh
    xt = full.reshape(M, H).astype(np.float32)
    c = {"G": G, "S": S, "x": xt}
    res = xt
    h, r1 = rmsnorm_forward(xt, w["input_ln"], si, cfg.eps)
    q = linear_bf16(h, w["q_proj"], si); k = linear_bf16(h, w["k_proj"], si); v = linear_bf16(h, w["v_proj"], si)
    c.update(rstd1=r1, h1=h, q=q, k=k)
    qn, rq = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], si, cfg.eps)
    kn, rk = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], si, cfg.eps)
    c.update(rstd_q=rq, rstd_k=rk)
    qh = qn.reshape(G, S, Hq, Dh).transpose(0, 2, 1, 3)          # NO rope
    kh = kn.reshape(G, S, Hkv, Dh).transpose(0, 2, 1, 3)
    vh = v.reshape(G, S, Hkv, Dh).transpose(0, 2, 1, 3)
    O, L = flash_attn_forward(qh, kh, vh, si, causal=True, return_lse=True)
    c.update(qh=qh, kh=kh, vh=vh, o_attn=O, L=L)
    o_tok = O.transpose(0, 2, 1, 3).reshape(M, qd)
    c["o_tok"] = o_tok
    xt = _bf(res + linear_bf16(o_tok, w["o_proj"], si))
    res = xt
    h, r2 = rmsnorm_forward(xt, w["post_ln"], si, cfg.eps)
    gate = linear_bf16(h, w["gate_proj"], si); up = linear_bf16(h, w["up_proj"], si)
    act = swiglu_forward(gate, up, si)
    c.update(x2=xt, rstd2=r2, h2=h, gate=gate, up=up, act=act)
    xt = _bf(res + linear_bf16(act, w["down_proj"], si))
    return xt.reshape(G, S, H), c


def _naive_nope_bwd(w, d_out, c, cfg):
    """TransformerLayer.backward minus rope_backward. d_out (G,S,H) → (d_in (G,S,H), grads)."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    G, S = c["G"], c["S"]; M = G * S
    g = {}
    d = d_out.reshape(M, H).astype(np.float32)
    d_res2 = d
    d_a, g["down_proj"] = linear_bf16_backward(d, c["act"], w["down_proj"], si)
    d_gate, d_up = swiglu_backward(c["gate"], c["up"], d_a, si)
    d_h2a, g["gate_proj"] = linear_bf16_backward(d_gate, c["h2"], w["gate_proj"], si)
    d_h2b, g["up_proj"]   = linear_bf16_backward(d_up,   c["h2"], w["up_proj"],   si)
    d_x2_mlp, g["post_ln"] = rmsnorm_backward(c["x2"], w["post_ln"], d_h2a + d_h2b, c["rstd2"], si)
    d_x2 = d_res2 + d_x2_mlp
    d_res1 = d_x2
    d_o_tok, g["o_proj"] = linear_bf16_backward(d_x2, c["o_tok"], w["o_proj"], si)
    d_o_hm = d_o_tok.reshape(G, S, Hq, Dh).transpose(0, 2, 1, 3)
    d_qh, d_kh, d_vh = flash_attn_backward(c["qh"], c["kh"], c["vh"], c["o_attn"], d_o_hm, c["L"], si)
    d_qn = d_qh.transpose(0, 2, 1, 3).reshape(M * Hq,  Dh)        # NO rope backward
    d_kn = d_kh.transpose(0, 2, 1, 3).reshape(M * Hkv, Dh)
    d_v  = d_vh.transpose(0, 2, 1, 3).reshape(M, Hkv * Dh)
    d_q, g["q_norm"] = rmsnorm_backward(c["q"].reshape(M * Hq,  Dh), w["q_norm"], d_qn, c["rstd_q"], si)
    d_k, g["k_norm"] = rmsnorm_backward(c["k"].reshape(M * Hkv, Dh), w["k_norm"], d_kn, c["rstd_k"], si)
    d_h1q, g["q_proj"] = linear_bf16_backward(d_q.reshape(M, Hq * Dh),  c["h1"], w["q_proj"], si)
    d_h1k, g["k_proj"] = linear_bf16_backward(d_k.reshape(M, Hkv * Dh), c["h1"], w["k_proj"], si)
    d_h1v, g["v_proj"] = linear_bf16_backward(d_v,                      c["h1"], w["v_proj"], si)
    d_x_attn, g["input_ln"] = rmsnorm_backward(c["x"], w["input_ln"], d_h1q + d_h1k + d_h1v, c["rstd1"], si)
    return (d_res1 + d_x_attn).reshape(G, S, H), g


def _case(G, Sp, Sc, top=True):
    cfg = Qwen3Config(); H = cfg.hidden
    L = TransformerLayer(cfg, seed=1); rng = np.random.default_rng(2)
    xp = (rng.standard_normal((Sp, H)) * 0.5).astype(np.float32)
    xs = (rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32)
    d_xs = (rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32)
    d_pe = np.zeros((Sp, H), np.float32) if top else (rng.standard_normal((Sp, H)) * 0.3).astype(np.float32)
    pa = PrefixGlobalAttn(cfg.n_heads, cfg.n_kv_heads, D, Sp, Sc, G)

    full = np.stack([np.concatenate([xp, xs[i]], 0) for i in range(G)])
    dfull = np.concatenate([np.tile(d_pe, (G, 1, 1)), d_xs], 1)

    e_g = e_xs = e_xp = 1e9; clean = False
    for _ in range(8):
        xpo, xso, c = prefix_layer_global(L.w, xp.copy(), xs.copy(), cfg, pa, si, return_cache=True)
        d_xp_in, d_xs_in, gP = prefix_layer_global_bwd(L.w, (G * d_pe).copy(), d_xs.copy(), c, cfg, pa, si)
        out, cn = _naive_nope_fwd(L.w, full, cfg)
        dxin, gN = _naive_nope_bwd(L.w, dfull, cn, cfg)
        if maxabs(xso, out[:, Sp:]) != 0.0: continue              # per-token glue churned → retry
        e_g  = max(rel(gP[k], gN[k]) for k in gP)
        e_xs = rel(d_xs_in, dxin[:, Sp:])
        e_xp = rel(d_xp_in, dxin[:, :Sp].sum(0))                 # prompt grad = Σ_G naive
        if max(e_g, e_xs, e_xp) < 0.02: clean = True; break
    tag = "top(d_xp=0)" if top else "lower(d_xp≠0)"
    print(f"  G={G} Sp={Sp} Sc={Sc} {tag:13s}: grad≤{e_g*100:.2f}%  d_xs≤{e_xs*100:.2f}%  "
          f"d_xp(ΣG)≤{e_xp*100:.2f}%  {'OK' if clean else 'FAIL'}")
    return clean


if __name__ == "__main__":
    print("Prefix-shared GLOBAL (NoPE) layer BACKWARD — grads + Σ_G prompt grad == naive NoPE layer")
    print("=" * 88)
    r = [_case(4, 128, 128, top=True), _case(6, 128, 64, top=True),
         _case(4, 128, 128, top=False), _case(4, 256, 64, top=False)]
    print("=" * 88)
    print("  ALL PASS (NoPE prefix layer bwd training-equivalent; prompt grad = Σ_G)" if all(r)
          else "  FAIL: " + str(r))
