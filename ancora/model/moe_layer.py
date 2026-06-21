"""
ancora/model/moe_layer.py — interleaved dense/MoE + local/global decoder layer.

DESIGN ARTIFACT FOR REVIEW (not yet kernel-optimized). Mirrors qwen3_layer.py's
correctness-first style: real kernels for the heavy GEMM/SwiGLU, host numpy for the
routing / reshapes / residual adds (the megakernel fuses those on-device later).

Architecture (MAI-Base-1 family, single-GPU "Config A", see CLAUDE.md discussion):
  - hidden 1024 · head_dim 128 · GQA 16 Q / 8 KV  (== Qwen3-0.6B attention shape)
  - 12 layers. FFN alternates  DENSE / MoE / DENSE / MoE ...  (first FFN is dense).
  - attention 5 local : 1 global. local = RoPE(base 1e4) + sliding-window 512;
    global = NoPE + full causal.
  - RMSNorm sandwich (input + post), QK-norm, no biases, tied embeddings.

THE FFN UP/DOWN DESIGN — uniform-square SwiGLU (the thing under review)
──────────────────────────────────────────────────────────────────────
Every FFN block in the model is the SAME 1024-square SwiGLU:
      up   :  H(1024) ─────────►  H(1024)     gate, up   (two of them: SwiGLU gate+value)
      down :  H(1024) ─────────►  H(1024)     down

  Why 1× (not the 4× I first suggested)?  SwiGLU's multiplicative gate already buys the
  expressiveness the ReLU-era 4× intermediate used to provide — the iso-param SwiGLU
  point is only 8/3≈2.67×. And in an INTERLEAVED dense/MoE stack the dense FFN isn't the
  main feature transform (the MoE layers are); it's connective tissue, so it can be small
  (MAI goes all the way to 1×).

  The elegant consequence — the dense FFN IS just an always-on expert:
      DENSE layer  = 1  always-on square SwiGLU expert   (a "shared expert", weight 1)
      MoE   layer  = 16 routed   square SwiGLU experts    (top-2 of 16)
  Same building block everywhere. The ONLY difference between a dense and an MoE layer is
  "1 always-on" vs "16 routed". Per-token active intermediate: dense 1024, MoE top-2 →
  2048. The MoE layer's CAPACITY comes from expert *count* (16×), not per-expert width.
  Kernel payoff: one square-SwiGLU GEMM shape; the MoE kernel is literally "the dense
  path, batched over experts, with a router in front".

  We DROP MAI's LatentMoE shared down-projection (a single GPU has no all-to-all to
  shrink) so experts read/write full hidden directly. No within-layer shared expert
  (the interleaved layout doesn't benefit — MAI §2.1); the dense LAYER plays that role.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataclasses import dataclass

import numpy as np
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # sets CUDA_PATH

# reuse the exact BF16 GEMM / helpers the Qwen3 layer uses (no duplication)
from ancora.model.qwen3_layer import _bf, linear_bf16, linear_bf16_backward
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.rope import rope_forward, rope_backward
from ancora.kernels.attention import flash_attn_forward, flash_attn_backward
from ancora.kernels.activation import swiglu_forward, swiglu_backward
from ancora.kernels.loss import GTM   # GEMM row-tile (=128); MoE expert groups pad to it


def _pad_rows(x, mult=GTM):
    """Pad a (M, *) array up to a multiple of `mult` rows with zeros → (xp, M).
    Zero rows produce zero GEMM output and contribute zero to any K-reduction, so a
    padded expert group is numerically identical to the unpadded one after slicing —
    exactly the 'expert capacity' tile-padding the real grouped-GEMM kernel will do."""
    M = x.shape[0]
    Mp = ((M + mult - 1) // mult) * mult
    if Mp == M:
        return x, M
    return np.concatenate([x, np.zeros((Mp - M,) + x.shape[1:], x.dtype)], 0), M


@dataclass
class MoEConfig:
    # ── shared ──
    hidden: int = 1024
    n_heads: int = 16            # Hq
    n_kv_heads: int = 8          # Hkv  (GQA group = 2)
    head_dim: int = 128
    eps: float = 1e-6
    vocab: int = 151936          # tied embedding (reuse Qwen3 tokenizer)
    n_layers: int = 12
    # ── attention: local/global periodic (Gemma-3 / MAI) ──
    period: int = 6              # 5 local : 1 global  → every 6th layer is global
    window: int = 512            # sliding-window size for LOCAL layers
    rope_theta_local: float = 1e4  # MAI uses base 1e4 on local layers (NOT Qwen3's 1e6)
    # global layers use NoPE (no positional encoding at all)
    # ── dense FFN: uniform-square (1×, == one always-on expert) ──
    dense_mult: int = 1          # intermediate = dense_mult * hidden = 1024 (== expert size)
    # ── MoE ──
    n_experts: int = 16          # E
    top_k: int = 2               # experts activated per token
    expert_mult: float = 1.0     # expert intermediate = expert_mult * hidden = 1024
    norm_topk: bool = True       # renormalize the top-k gate weights to sum to 1
    first_ffn_dense: bool = True # layer 0's FFN is dense (DeepSeek/Kimi convention)

    @property
    def dense_inter(self):  return int(self.dense_mult * self.hidden)
    @property
    def expert_inter(self): return int(self.expert_mult * self.hidden)


def layer_schedule(cfg: MoEConfig):
    """Return a list of (is_global, ffn_is_dense) per layer.

      attn:  global every `period`-th layer (the 6th of each 5-local block).
      ffn :  dense / moe alternating, layer 0 dense.

    L=12, period=6, first_ffn_dense=True →
      idx : 0   1   2   3   4   5   6   7   8   9  10  11
      attn: L   L   L   L   L   G   L   L   L   L   L   G
      ffn : D   M   D   M   D   M   D   M   D   M   D   M
    """
    sched = []
    for i in range(cfg.n_layers):
        is_global = ((i + 1) % cfg.period == 0)
        ffn_dense = ((i % 2) == 0) if cfg.first_ffn_dense else ((i % 2) == 1)
        sched.append((is_global, ffn_dense))
    return sched


# ════════════════════════════════════════════════════════════════════════════
#  FFN variant 1 — DENSE  (1-4-1 SwiGLU, identical wiring to Qwen3, just wider)
# ════════════════════════════════════════════════════════════════════════════
class DenseFFN:
    """h(M,H) → SwiGLU through 1×H intermediate → (M,H). Every token, no routing.
    Structurally IDENTICAL to one MoE expert — an always-on, always-selected expert."""
    def __init__(self, cfg: MoEConfig, rng):
        H, I = cfg.hidden, cfg.dense_inter   # I == H in the uniform-square design
        def W(shape, s=0.02): return _bf((rng.standard_normal(shape) * s).astype(np.float32))
        self.w = {
            "gate_proj": W((H, I)),     # up-projection (gate branch)
            "up_proj":   W((H, I)),     # up-projection (value branch)
            "down_proj": W((I, H)),     # down-projection
        }

    def forward(self, h, stream_int):
        gate = linear_bf16(h, self.w["gate_proj"], stream_int)   # (M, H)
        up   = linear_bf16(h, self.w["up_proj"],   stream_int)   # (M, H)
        act  = swiglu_forward(gate, up, stream_int)              # (M, H)
        out  = linear_bf16(act, self.w["down_proj"], stream_int) # (M, H)
        return out, {"gate": gate, "up": up, "act": act}

    def backward(self, d_out, h, cache, stream_int):
        g = {}
        d_a, g["down_proj"] = linear_bf16_backward(d_out, cache["act"], self.w["down_proj"], stream_int)
        d_gate, d_up = swiglu_backward(cache["gate"], cache["up"], d_a, stream_int)
        d_hg, g["gate_proj"] = linear_bf16_backward(d_gate, h, self.w["gate_proj"], stream_int)
        d_hu, g["up_proj"]   = linear_bf16_backward(d_up,   h, self.w["up_proj"],   stream_int)
        return d_hg + d_hu, g


# ════════════════════════════════════════════════════════════════════════════
#  FFN variant 2 — MoE  (router → top-k → per-expert square SwiGLU → weighted sum)
# ════════════════════════════════════════════════════════════════════════════
class MoEFFN:
    """h(M,H) → route to top_k of E experts, each a 1×H SwiGLU, gate-weighted sum.

    Reference path: host-numpy routing + a python loop over experts calling the BF16
    GEMM/SwiGLU kernels per expert group. The PERF path replaces the loop with one
    grouped/segmented GEMM (sort tokens by expert id, fixed-order accumulation, no
    atomics → batch-invariant, see CLAUDE.md rule on stable-sort top-k). Same numbers.
    """
    def __init__(self, cfg: MoEConfig, rng):
        H, Ie, E = cfg.hidden, cfg.expert_inter, cfg.n_experts
        self.cfg = cfg
        def W(shape, s=0.02): return _bf((rng.standard_normal(shape) * s).astype(np.float32))
        self.w = {
            "router":    W((H, E), s=0.02),        # token → expert logits  (H,E)
            "gate_proj": W((E, H, Ie)),            # per-expert up (gate)    (E,H,Ie)
            "up_proj":   W((E, H, Ie)),            # per-expert up (value)   (E,H,Ie)
            "down_proj": W((E, Ie, H)),            # per-expert down         (E,Ie,H)
        }

    # ── gating: softmax over ALL experts, take top-k, (optionally) renormalize ──
    def _route(self, h, stream_int):
        cfg = self.cfg
        # Router runs in fp32 (it's tiny, H→E=16; routing is precision-sensitive — MAI
        # keeps it fp32 — and N=E=16 isn't a valid kernel GEMM-N anyway). Deterministic.
        logits = h.astype(np.float32) @ self.w["router"].astype(np.float32)   # (M, E)
        z = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)          # (M, E)
        # stable sort → ties break to the lower expert id → top-k is batch-invariant
        topi = np.argsort(-probs, axis=1, kind="stable")[:, :cfg.top_k]       # (M, k)
        topw = np.take_along_axis(probs, topi, axis=1)                        # (M, k)
        if cfg.norm_topk:
            topw = topw / topw.sum(axis=1, keepdims=True)
        return logits, probs, topi, topw

    def forward(self, h, stream_int):
        cfg = self.cfg; E, H = cfg.n_experts, cfg.hidden
        M = h.shape[0]
        logits, probs, topi, topw = self._route(h, stream_int)

        out = np.zeros((M, H), np.float32)
        # token→slot bookkeeping so backward can reproduce the exact grouping
        groups = {}                                  # expert e → (rows, gate_w, padded acts)
        for e in range(E):
            sel = np.where(topi == e)                # (rows, k-slot) where expert e was picked
            rows = sel[0]                            # token rows routed to expert e (unique)
            if rows.size == 0:
                groups[e] = (rows, None, None); continue
            w_e = topw[sel]                          # (Me,) gate weight for these tokens
            Me  = rows.size
            hep, _ = _pad_rows(h[rows])              # (Mp, H)  pad the group to a 128-tile
            ge  = linear_bf16(hep, self.w["gate_proj"][e], stream_int)   # (Mp, Ie)
            ue  = linear_bf16(hep, self.w["up_proj"][e],   stream_int)   # (Mp, Ie)
            ae  = swiglu_forward(ge, ue, stream_int)                     # (Mp, Ie)
            oe  = linear_bf16(ae, self.w["down_proj"][e], stream_int)    # (Mp, H)
            out[rows] += w_e[:, None] * oe[:Me]      # gate-weight AFTER down (SwiGLU nonlinear)
            groups[e] = (rows, w_e, {"hep": hep, "ge": ge, "ue": ue, "ae": ae,
                                     "oe": oe[:Me], "Me": Me})

        cache = {"logits": logits, "probs": probs, "topi": topi, "topw": topw, "groups": groups}
        return out, cache

    def backward(self, d_out, h, cache, stream_int):
        """v1 — validate forward dims first; this is the documented gradient flow.
        d_out:(M,H). Two paths feed dh:
          (1) through each expert f_e, scaled by its gate weight w_e;
          (2) through the gate weights w_e = softmax/top-k(router(h)) — the router grad.
        """
        cfg = self.cfg; E, H, Ie = cfg.n_experts, cfg.hidden, cfg.expert_inter
        M = h.shape[0]
        g = {"router": np.zeros_like(self.w["router"]),
             "gate_proj": np.zeros_like(self.w["gate_proj"]),
             "up_proj":   np.zeros_like(self.w["up_proj"]),
             "down_proj": np.zeros_like(self.w["down_proj"])}
        d_h = np.zeros((M, H), np.float32)
        d_w = np.zeros((M, cfg.n_experts), np.float32)   # grad wrt each token's gate weights

        groups = cache["groups"]
        for e in range(E):
            rows, w_e, sub = groups[e]
            if rows.size == 0: continue
            Me = sub["Me"]
            do_e = d_out[rows]                                   # (Me, H)
            # grad wrt the gate weight: <d_out, expert_output>
            d_w[rows, e] = np.sum(do_e * sub["oe"], axis=1)
            # grad through the expert (output scaled by w_e); pad d_oe back to the tile
            d_oep, _ = _pad_rows(w_e[:, None] * do_e)            # (Mp, H)
            d_ae, g["down_proj"][e] = linear_bf16_backward(d_oep, sub["ae"], self.w["down_proj"][e], stream_int)
            d_ge, d_ue = swiglu_backward(sub["ge"], sub["ue"], d_ae, stream_int)
            d_hg, g["gate_proj"][e] = linear_bf16_backward(d_ge, sub["hep"], self.w["gate_proj"][e], stream_int)
            d_hu, g["up_proj"][e]   = linear_bf16_backward(d_ue, sub["hep"], self.w["up_proj"][e],   stream_int)
            d_h[rows] += (d_hg + d_hu)[:Me]
        # router grad: d_w (grad wrt selected gate weights) → renorm+softmax → logits →
        # W_router. Router is fp32 host (mirror of _route), so its grad is host too.
        d_logits = self._gate_backward(cache, d_w)               # (M, E)
        g["router"] = h.astype(np.float32).T @ d_logits          # (H, E)
        d_h += d_logits @ self.w["router"].astype(np.float32).T  # (M, H)
        return d_h, g

    def _gate_backward(self, cache, d_w):
        """Backprop d(loss)/d(gate weights) → d(logits). Handles top-k renorm + softmax.
        Only the top-k entries of d_w are non-zero. (host numpy; small, per-row.)"""
        cfg = self.cfg
        probs, topi, topw = cache["probs"], cache["topi"], cache["topw"]
        M, E = probs.shape
        d_logits = np.zeros((M, E), np.float32)
        # gather d wrt the (pre-renorm) selected softmax probs
        d_sel = np.take_along_axis(d_w, topi, axis=1).astype(np.float32)   # (M, k)
        if cfg.norm_topk:
            raw = np.take_along_axis(probs, topi, axis=1)                  # (M, k)
            s = raw.sum(axis=1, keepdims=True)
            # d(raw_i)/ from w_i = raw_i / s : Jacobian of the normalize
            dot = (d_sel * (raw / s)).sum(axis=1, keepdims=True)
            d_sel = (d_sel - dot) / s
        # scatter back to a full (M,E) grad wrt softmax probs, then softmax Jacobian
        d_probs = np.zeros((M, E), np.float32)
        np.put_along_axis(d_probs, topi, d_sel, axis=1)
        dot = (d_probs * probs).sum(axis=1, keepdims=True)
        d_logits = probs * (d_probs - dot)                                # softmax backward
        return d_logits

    def aux_loss_stats(self, cache):
        """GShard load-balancing aux: returns (f_e, P_e) per expert for the training
        loop to form  E * Σ_e f_e * P_e.  f_e = fraction of tokens routed to e,
        P_e = mean router prob for e (aggregated GLOBALLY across the batch — CLAUDE.md)."""
        probs, topi = cache["probs"], cache["topi"]
        M, E = probs.shape
        f = np.bincount(topi.reshape(-1), minlength=E).astype(np.float32) / M
        P = probs.mean(axis=0)
        return f, P


# ════════════════════════════════════════════════════════════════════════════
#  DECODER LAYER — attention (local|global) + one FFN (dense|moe)
# ════════════════════════════════════════════════════════════════════════════
class MoEDecoderLayer:
    """One block: RMSNorm-sandwich attention (local/global) then dense-or-MoE FFN."""
    def __init__(self, cfg: MoEConfig, is_global: bool, ffn_dense: bool, seed: int = 0, grouped: bool = False):
        self.cfg, self.is_global, self.ffn_dense = cfg, is_global, ffn_dense
        rng = np.random.default_rng(seed)
        H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        qd, kd = Hq * Dh, Hkv * Dh
        def W(shape, s=0.02): return _bf((rng.standard_normal(shape) * s).astype(np.float32))
        def G(n): return _bf((1.0 + rng.standard_normal(n) * 0.05).astype(np.float32))
        self.attn = {
            "input_ln": G(H),
            "q_proj": W((H, qd)), "k_proj": W((H, kd)), "v_proj": W((H, kd)),
            "q_norm": G(Dh), "k_norm": G(Dh),
            "o_proj": W((qd, H)),
            "post_ln": G(H),
        }
        if ffn_dense:
            self.ffn = DenseFFN(cfg, rng)
        elif grouped:
            from ancora.kernels.moe import GroupedMoEFFN          # device grouped kernel (no churn)
            self.ffn = GroupedMoEFFN(MoEFFN(cfg, rng).w, cfg.top_k)
        else:
            self.ffn = MoEFFN(cfg, rng)

    # ── attention sublayer (parameterized local vs global) ──
    def _attention(self, xt, B, S, stream_int):
        cfg = self.cfg; w = self.attn
        Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        M = B * S
        residual = xt
        h, r1 = rmsnorm_forward(xt, w["input_ln"], stream_int, cfg.eps)
        q = linear_bf16(h, w["q_proj"], stream_int)
        k = linear_bf16(h, w["k_proj"], stream_int)
        v = linear_bf16(h, w["v_proj"], stream_int)
        qn, rq = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], stream_int, cfg.eps)
        kn, rk = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], stream_int, cfg.eps)
        qh = qn.reshape(B, S, Hq,  Dh).transpose(0, 2, 1, 3)
        kh = kn.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
        vh = v.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
        if not self.is_global:
            # LOCAL: RoPE (base 1e4) + sliding window. NoPE on global → skip RoPE.
            qh = rope_forward(qh, stream_int, cfg.rope_theta_local)
            kh = rope_forward(kh, stream_int, cfg.rope_theta_local)
        # LOCAL → sliding-window causal (query i sees keys (i-window, i]); GLOBAL → full causal.
        window = 0 if self.is_global else cfg.window
        o, L = flash_attn_forward(qh, kh, vh, stream_int, causal=True, return_lse=True, window=window)
        o_hm = o                                                  # head-major (B,Hq,Sq,D) for attn bwd
        o = o.transpose(0, 2, 1, 3).reshape(M, Hq * Dh)           # token-major for o_proj
        attn = linear_bf16(o, w["o_proj"], stream_int)
        out = _bf(residual + attn)
        cache = {"x": xt, "rstd1": r1, "h1": h, "q": q, "k": k, "rstd_q": rq, "rstd_k": rk,
                 "qh": qh, "kh": kh, "vh": vh, "o_attn": o_hm, "L": L, "o_tok": o, "window": window}
        return out, cache

    def forward(self, x, stream_int, return_cache=False):
        cfg = self.cfg
        B, S, H = x.shape; M = B * S
        xt = x.reshape(M, H).astype(np.float32)
        xt, ac = self._attention(xt, B, S, stream_int)
        # ── FFN sublayer (RMSNorm-sandwich residual) ──
        residual = xt
        h, r2 = rmsnorm_forward(xt, self.attn["post_ln"], stream_int, cfg.eps)
        ff, fc = self.ffn.forward(h, stream_int)
        xt = _bf(residual + ff)
        c = {"B": B, "S": S, "attn": ac, "x2": residual, "rstd2": r2, "h2": h, "ffn": fc}
        out = xt.reshape(B, S, H)
        return (out, c) if return_cache else out

    def backward(self, d_out, cache, stream_int):
        """Reverse of forward. Mirror of qwen3_layer.backward with (a) self.ffn.backward
        (dense or MoE) swapped in for the SwiGLU block, (b) windowed attention backward for
        local layers, (c) NoPE (skip rope_backward) for global layers. Returns (d_x, grads);
        grads has attention weights flat + the FFN grads under g["ffn"]."""
        cfg = self.cfg; c = cache; ac = c["attn"]; w = self.attn
        Hq, Hkv, Dh, H = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden
        B, S = c["B"], c["S"]; M = B * S
        g = {}
        d = d_out.reshape(M, H).astype(np.float32)

        # ── FFN backward (dense or MoE) ──
        d_res2 = d                                              # out = x2 + ffn
        d_h2, g["ffn"] = self.ffn.backward(d, c["h2"], c["ffn"], stream_int)
        d_x2_mlp, g["post_ln"] = rmsnorm_backward(c["x2"], w["post_ln"], d_h2, c["rstd2"], stream_int)
        d_x2 = d_res2 + d_x2_mlp

        # ── Attention backward ──
        d_res1 = d_x2                                           # x2 = x + attn
        d_o_tok, g["o_proj"] = linear_bf16_backward(d_x2, ac["o_tok"], w["o_proj"], stream_int)
        d_o_hm = d_o_tok.reshape(B, S, Hq, Dh).transpose(0, 2, 1, 3)
        d_qh, d_kh, d_vh = flash_attn_backward(ac["qh"], ac["kh"], ac["vh"], ac["o_attn"],
                                               d_o_hm, ac["L"], stream_int, window=ac["window"])
        if not self.is_global:                                 # undo RoPE (local); global is NoPE
            d_qn_hm = rope_backward(d_qh, stream_int, cfg.rope_theta_local)
            d_kn_hm = rope_backward(d_kh, stream_int, cfg.rope_theta_local)
        else:
            d_qn_hm, d_kn_hm = d_qh, d_kh
        d_qn = d_qn_hm.transpose(0, 2, 1, 3).reshape(M * Hq,  Dh)
        d_kn = d_kn_hm.transpose(0, 2, 1, 3).reshape(M * Hkv, Dh)
        d_v  = d_vh.transpose(0, 2, 1, 3).reshape(M, Hkv * Dh)
        d_q, g["q_norm"] = rmsnorm_backward(ac["q"].reshape(M * Hq,  Dh), w["q_norm"], d_qn, ac["rstd_q"], stream_int)
        d_k, g["k_norm"] = rmsnorm_backward(ac["k"].reshape(M * Hkv, Dh), w["k_norm"], d_kn, ac["rstd_k"], stream_int)
        d_q = d_q.reshape(M, Hq * Dh); d_k = d_k.reshape(M, Hkv * Dh)
        d_h1q, g["q_proj"] = linear_bf16_backward(d_q, ac["h1"], w["q_proj"], stream_int)
        d_h1k, g["k_proj"] = linear_bf16_backward(d_k, ac["h1"], w["k_proj"], stream_int)
        d_h1v, g["v_proj"] = linear_bf16_backward(d_v, ac["h1"], w["v_proj"], stream_int)
        d_h1 = d_h1q + d_h1k + d_h1v
        d_x_attn, g["input_ln"] = rmsnorm_backward(ac["x"], w["input_ln"], d_h1, ac["rstd1"], stream_int)
        d_x = d_res1 + d_x_attn
        return d_x.reshape(B, S, H), g
