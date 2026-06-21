"""
ancora/rl/prefix_grpo.py — prefix-shared GRPO training step (Prefix Grouper wiring): forward + backward.

GRPO makes G completions per prompt sharing the prompt PREFIX; the naive training pass (model on the
G replicated [prompt, completion_i]) encodes the prompt G×. Here the prompt is laid out ONCE and the
G completions' suffixes attend it (attention._attn_fwd_prefix). Token budget: Sp + G·Sc (vs G·(Sp+Sc)).

TRAINING-EQUIVALENT (Prefix Grouper): the per-token ops are per-row; attention is bitwise-equal to
standard attn on [P,s_i]; RoPE puts the suffix at positions Sp..; and the prompt's gradient is the
Σ-over-G cross term (attention._attn_bwd_dkdv_prefix) — so processing the prompt ONCE with that summed
gradient yields the SAME weight grads as the naive G copies. ⇒ completion logprobs + all grads match
⇒ ratio=1, at G·(Sp+Sc)→Sp+G·Sc tokens.

  lp, grads = prefix_grpo_loss_backward(model, prompt_ids (Sp,), comp_ids (G,Sc), labels, adv, si)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.qwen3_layer import linear_bf16, linear_bf16_backward, _bf
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.activation import swiglu_forward, swiglu_backward
from ancora.kernels.rope import (_rope_fwd, _rope_bwd, build_cos_sin, rope_forward, rope_backward,
                                  f32_to_bf16_bits as _f32bf_rne, RTM)
from ancora.kernels.attention import (flash_attn_forward, flash_attn_forward_prefix,
                                       flash_attn_backward_prefix, _GpuArray, _f32_to_bf16_bits as _f32bf)
from ancora.kernels.loss import linear_ce

_bits2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def _rope_seg(x, offset, base, si, kernel):
    """RoPE (fwd or bwd) a head-major (n_heads, Sc, Dh) tensor at positions [offset, offset+Sc)."""
    Hh, Sc, Dh = x.shape
    cos, sin = build_cos_sin(offset + Sc, Dh, base)
    cos = np.ascontiguousarray(cos[offset:offset + Sc]); sin = np.ascontiguousarray(sin[offset:offset + Sc])
    gx = _GpuArray(_f32bf_rne(x.reshape(Hh * Sc, Dh)))   # RNE — MUST match rope_forward (truncation → 1-ULP drift)
    gc = _GpuArray(cos); gs = _GpuArray(sin)
    gy = _GpuArray.zeros((Hh * Sc, Dh), np.uint16)
    ct.launch(si, (Sc // RTM, Hh, 1), kernel, (gx, gc, gs, gy, Sc // RTM, Dh // 2))
    cudart.cudaStreamSynchronize(si)
    y = _bits2f(gy.to_numpy()).reshape(Hh, Sc, Dh)
    for g in (gx, gc, gs, gy): g.free()
    return y


def _rope_off(x, offset, base, si):     return _rope_seg(x, offset, base, si, _rope_fwd)
def _rope_off_bwd(dy, offset, base, si): return _rope_seg(dy, offset, base, si, _rope_bwd)
def _rope_p(x_tok, si, theta):
    """RoPE prompt q/k (Sp,H_,Dh) token-major at positions 0..Sp-1 → head-major (H_,Sp,Dh)."""
    return rope_forward(x_tok.transpose(1, 0, 2)[None], si, theta)[0]


def _prefix_layer(w, xp, xs, cfg, si):
    """One prefix-shared decoder layer. xp (Sp,H), xs (G,Sc,H) → (xp',xs',cache)."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    Sp = xp.shape[0]; G, Sc, _ = xs.shape; theta = cfg.rope_theta; qd = Hq * Dh
    xall = np.concatenate([xp, xs.reshape(G * Sc, H)], 0); M = xall.shape[0]
    c = dict(xall=xall, Sp=Sp, G=G, Sc=Sc)

    res = xall
    h1, r1 = rmsnorm_forward(xall, w["input_ln"], si, cfg.eps)
    q = linear_bf16(h1, w["q_proj"], si); k = linear_bf16(h1, w["k_proj"], si); v = linear_bf16(h1, w["v_proj"], si)
    qn, rq = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], si, cfg.eps)
    kn, rk = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], si, cfg.eps)
    c.update(r1=r1, h1=h1, q=q, k=k, rq=rq, rk=rk)
    qn = qn.reshape(M, Hq, Dh); kn = kn.reshape(M, Hkv, Dh); vh = v.reshape(M, Hkv, Dh)
    qr_p = _rope_p(qn[:Sp], si, theta); kr_p = _rope_p(kn[:Sp], si, theta); vh_p = vh[:Sp].transpose(1, 0, 2)
    qs = qn[Sp:].reshape(G, Sc, Hq, Dh).transpose(0, 2, 1, 3).reshape(G * Hq, Sc, Dh)
    ks = kn[Sp:].reshape(G, Sc, Hkv, Dh).transpose(0, 2, 1, 3).reshape(G * Hkv, Sc, Dh)
    qr_s = _rope_off(qs, Sp, theta, si).reshape(G, Hq, Sc, Dh)
    kr_s = _rope_off(ks, Sp, theta, si).reshape(G, Hkv, Sc, Dh)
    vh_s = vh[Sp:].reshape(G, Sc, Hkv, Dh).transpose(0, 2, 1, 3)
    Op, Os, Lp, Ls = flash_attn_forward_prefix(qr_p, kr_p, vh_p, qr_s, kr_s, vh_s, si)
    c.update(qr_p=qr_p, kr_p=kr_p, vh_p=vh_p, qr_s=qr_s, kr_s=kr_s, vh_s=vh_s, Op=Op, Os=Os, Lp=Lp, Ls=Ls)
    o_tok = np.concatenate([Op.transpose(1, 0, 2).reshape(Sp, qd),
                            Os.transpose(0, 2, 1, 3).reshape(G * Sc, qd)], 0)
    c["o_tok"] = o_tok
    xall = _bf(res + linear_bf16(o_tok, w["o_proj"], si))

    res = xall
    h2, r2 = rmsnorm_forward(xall, w["post_ln"], si, cfg.eps)
    gate = linear_bf16(h2, w["gate_proj"], si); up = linear_bf16(h2, w["up_proj"], si)
    act = swiglu_forward(gate, up, si)
    c.update(x2=xall, r2=r2, h2=h2, gate=gate, up=up, act=act)
    xall = _bf(res + linear_bf16(act, w["down_proj"], si))
    return xall[:Sp], xall[Sp:].reshape(G, Sc, H), c


def _prefix_layer_bwd(w, d_xp, d_xs, c, cfg, si):
    """Reverse of _prefix_layer. d_xp (Sp,H), d_xs (G,Sc,H) → (d_xp_in, d_xs_in, grads)."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    Sp, G, Sc = c["Sp"], c["G"], c["Sc"]; theta = cfg.rope_theta; qd = Hq * Dh; M = Sp + G * Sc
    g = {}
    d = np.concatenate([d_xp.reshape(Sp, H), d_xs.reshape(G * Sc, H)], 0)

    # MLP backward (all tokens)
    d_res2 = d
    d_a, g["down_proj"] = linear_bf16_backward(d, c["act"], w["down_proj"], si)
    d_gate, d_up = swiglu_backward(c["gate"], c["up"], d_a, si)
    d_h2a, g["gate_proj"] = linear_bf16_backward(d_gate, c["h2"], w["gate_proj"], si)
    d_h2b, g["up_proj"]   = linear_bf16_backward(d_up,   c["h2"], w["up_proj"],   si)
    d_x2_mlp, g["post_ln"] = rmsnorm_backward(c["x2"], w["post_ln"], d_h2a + d_h2b, c["r2"], si)
    d_x2 = d_res2 + d_x2_mlp

    # attention backward (prefix-shared)
    d_res1 = d_x2
    d_o_tok, g["o_proj"] = linear_bf16_backward(d_x2, c["o_tok"], w["o_proj"], si)
    d_o_p = d_o_tok[:Sp].reshape(Sp, Hq, Dh).transpose(1, 0, 2)                  # (Hq, Sp, Dh)
    d_o_s = d_o_tok[Sp:].reshape(G, Sc, Hq, Dh).transpose(0, 2, 1, 3)           # (G, Hq, Sc, Dh)
    dQp, dKp, dVp, dQs, dKs, dVs = flash_attn_backward_prefix(
        c["qr_p"], c["kr_p"], c["vh_p"], c["qr_s"], c["kr_s"], c["vh_s"],
        c["Op"], c["Os"], c["Lp"], c["Ls"], d_o_p, d_o_s, si)
    # undo RoPE: prompt at 0.., suffix at Sp..
    dqn_p = rope_backward(dQp[None], si, theta)[0]; dkn_p = rope_backward(dKp[None], si, theta)[0]
    dqn_s = _rope_off_bwd(dQs.reshape(G * Hq, Sc, Dh), Sp, theta, si).reshape(G, Hq, Sc, Dh)
    dkn_s = _rope_off_bwd(dKs.reshape(G * Hkv, Sc, Dh), Sp, theta, si).reshape(G, Hkv, Sc, Dh)
    # reassemble token-major (prompt rows then suffix rows)
    d_qn = np.concatenate([dqn_p.transpose(1, 0, 2).reshape(Sp, Hq, Dh),
                           dqn_s.transpose(0, 2, 1, 3).reshape(G * Sc, Hq, Dh)], 0).reshape(M * Hq, Dh)
    d_kn = np.concatenate([dkn_p.transpose(1, 0, 2).reshape(Sp, Hkv, Dh),
                           dkn_s.transpose(0, 2, 1, 3).reshape(G * Sc, Hkv, Dh)], 0).reshape(M * Hkv, Dh)
    d_v  = np.concatenate([dVp.transpose(1, 0, 2).reshape(Sp, Hkv, Dh),
                           dVs.transpose(0, 2, 1, 3).reshape(G * Sc, Hkv, Dh)], 0).reshape(M, Hkv * Dh)
    d_q, g["q_norm"] = rmsnorm_backward(c["q"].reshape(M * Hq,  Dh), w["q_norm"], d_qn, c["rq"], si)
    d_k, g["k_norm"] = rmsnorm_backward(c["k"].reshape(M * Hkv, Dh), w["k_norm"], d_kn, c["rk"], si)
    d_h1q, g["q_proj"] = linear_bf16_backward(d_q.reshape(M, qd),       c["h1"], w["q_proj"], si)
    d_h1k, g["k_proj"] = linear_bf16_backward(d_k.reshape(M, Hkv * Dh), c["h1"], w["k_proj"], si)
    d_h1v, g["v_proj"] = linear_bf16_backward(d_v,                       c["h1"], w["v_proj"], si)
    d_x_attn, g["input_ln"] = rmsnorm_backward(c["xall"], w["input_ln"], d_h1q + d_h1k + d_h1v, c["r1"], si)
    d_x = d_res1 + d_x_attn
    return d_x[:Sp], d_x[Sp:].reshape(G, Sc, H), g


def prefix_group_forward(model, prompt_ids, comp_ids, si, return_cache=False):
    """prompt_ids (Sp,), comp_ids (G,Sc) → completion hidden (G*Sc, H) after the final RMSNorm."""
    cfg = model.cfg; H = cfg.hidden; G, Sc = comp_ids.shape
    xp = model.embed[prompt_ids].astype(np.float32)
    xs = model.embed[comp_ids.reshape(-1)].reshape(G, Sc, H).astype(np.float32)
    caches = []
    for l in model.layers:
        xp, xs, c = _prefix_layer(l.w, xp, xs, cfg, si); caches.append(c)
    h, rf = rmsnorm_forward(xs.reshape(G * Sc, H), model.final_norm, si, cfg.eps)
    if return_cache:
        return h, dict(caches=caches, x_pre=xs.reshape(G * Sc, H), rstd_f=rf)
    return h


def prefix_grpo_loss_backward(model, prompt_ids, comp_ids, labels, advantage, si):
    """Full prefix-shared GRPO step backward. Returns (logprob (G*Sc,), grads matching model.params()).
    Loss is on the COMPLETION tokens only (the generated suffix)."""
    cfg = model.cfg; H = cfg.hidden; G, Sc = comp_ids.shape
    h, cache = prefix_group_forward(model, prompt_ids, comp_ids, si, return_cache=True)
    adv = advantage.astype(np.float32)
    lp, dhidden, dW_head = linear_ce(h, model.lm_head, labels, si, advantage=adv)
    d_xpre, d_final = rmsnorm_backward(cache["x_pre"], model.final_norm, dhidden, cache["rstd_f"], si)

    grads = {"final_norm": d_final, "lm_head": dW_head}
    d_xp = np.zeros((prompt_ids.shape[0], H), np.float32)                       # no direct loss on prompt
    d_xs = d_xpre.reshape(G, Sc, H)
    for i in reversed(range(len(model.layers))):
        d_xp, d_xs, lg = _prefix_layer_bwd(model.layers[i].w, d_xp, d_xs, cache["caches"][i], cfg, si)
        for n, gg in lg.items():
            grads[f"layer{i}.{n}"] = gg
    # embed grad: prompt (gradient = Σ_G via the cross term) once + completion tokens
    d_embed = np.zeros_like(model.embed)
    np.add.at(d_embed, prompt_ids, d_xp)
    np.add.at(d_embed, comp_ids.reshape(-1), d_xs.reshape(G * Sc, H))
    grads["embed"] = d_embed
    return lp, grads
