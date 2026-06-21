"""
ancora/rl/prefix_resident.py — DEVICE-RESIDENT prefix-shared attention for the MoE GLOBAL (NoPE) layers.

The host flash_attn_forward_prefix allocs/frees per call → the documented alloc-churn (what limited the
host GRPO step test). This keeps the prompt+suffix Q/K/V/O/L in PERSISTENT _DBuf buffers and only
uploads (HtoDAsync on si) + launches the kernels — no per-call alloc ⇒ churn-free / bitwise-deterministic.

GLOBAL layers are NoPE (no RoPE — user) so this is the cleanest, biggest-win prefix-sharing target:
full attention over the WHOLE prompt, encoded once, G completions cross-attend it. Same kernels
(_attn_fwd on the prompt, _attn_fwd_prefix on the suffixes) → bitwise-equal to the host helper.

  pa = PrefixGlobalAttn(Hq, Hkv, D, Sp, Sc, G)
  Op, Os = pa.forward(pq, pk, pv, sq, sk, sv, si)   # prompt-self O + per-completion suffix O
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa: F401

from ancora.model.resident import _DBuf
from ancora.kernels.attention import (_attn_fwd, _attn_fwd_prefix, _attn_bwd_dq, _attn_bwd_dkdv,
                                       _attn_bwd_dq_prefix, _attn_bwd_dkdv_prefix,
                                       _attn_fwd_win, _attn_bwd_dq_win, _attn_bwd_dkdv_win,
                                       _attn_fwd_prefix_win, _attn_bwd_dq_prefix_win,
                                       _attn_bwd_dkdv_prefix_win, BQ, BKV,
                                       _f32_to_bf16_bits as _f32bf)


def _up(buf, host, si):
    """Upload host array into the front of a persistent buffer on stream si (no alloc)."""
    h = np.ascontiguousarray(host)
    cdrv.cuMemcpyHtoDAsync(buf.ptr, h, h.nbytes, si)
    return h                                            # keep alive until the caller syncs


def prefix_layer(w, xp, xs, cfg, pa, si, ffn=None, theta=None, return_cache=False):
    """One prefix-shared decoder layer (the MoE-model building block). xp (Sp,H) prompt,
    xs (G,Sc,H) suffixes → (xp', xs'). Per-token ops (norm/GEMM/FFN) on [prompt;G·suffix];
    attention = the CHURN-FREE resident PrefixGlobalAttn. The prompt is encoded ONCE.
      theta=None → GLOBAL: NoPE (no RoPE) + pa full causal (window=0).
      theta=…    → LOCAL : RoPE prompt@0.. / suffix@Sp.. (offset, RNE bf16) + pa sliding window.
      ffn=None → dense SwiGLU from w (gate/up/down_proj); ffn=GroupedMoEFFN → routed MoE FFN
      (per-token router + grouped GEMM are row-independent ⇒ prefix-sharing stays bitwise)."""
    from ancora.model.qwen3_layer import linear_bf16, _bf
    from ancora.kernels.norm import rmsnorm_forward
    from ancora.kernels.activation import swiglu_forward
    from ancora.rl.prefix_grpo import _rope_p, _rope_off
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    Sp = xp.shape[0]; G, Sc, _ = xs.shape; qd = Hq * Dh
    xall = np.concatenate([xp, xs.reshape(G * Sc, H)], 0); M = xall.shape[0]
    c = dict(xall=xall, Sp=Sp, G=G, Sc=Sc)

    res = xall
    h1, r1 = rmsnorm_forward(xall, w["input_ln"], si, cfg.eps)
    q = linear_bf16(h1, w["q_proj"], si); k = linear_bf16(h1, w["k_proj"], si); v = linear_bf16(h1, w["v_proj"], si)
    qn, rq = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], si, cfg.eps)
    kn, rk = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], si, cfg.eps)
    c.update(r1=r1, h1=h1, q=q, k=k, rq=rq, rk=rk)
    qn = qn.reshape(M, Hq, Dh); kn = kn.reshape(M, Hkv, Dh); vh = v.reshape(M, Hkv, Dh)
    # head-major split prompt / suffix; LOCAL ropes q/k (prompt@0.., suffix@Sp..), GLOBAL is NoPE
    vp = vh[:Sp].transpose(1, 0, 2)
    vs = vh[Sp:].reshape(G, Sc, Hkv, Dh).transpose(0, 2, 1, 3)
    if theta is None:
        qp = qn[:Sp].transpose(1, 0, 2); kp = kn[:Sp].transpose(1, 0, 2)
        qs = qn[Sp:].reshape(G, Sc, Hq, Dh).transpose(0, 2, 1, 3)
        ks = kn[Sp:].reshape(G, Sc, Hkv, Dh).transpose(0, 2, 1, 3)
    else:
        qp = _rope_p(qn[:Sp], si, theta); kp = _rope_p(kn[:Sp], si, theta)
        qs = _rope_off(qn[Sp:].reshape(G, Sc, Hq, Dh).transpose(0, 2, 1, 3).reshape(G * Hq, Sc, Dh),
                       Sp, theta, si).reshape(G, Hq, Sc, Dh)
        ks = _rope_off(kn[Sp:].reshape(G, Sc, Hkv, Dh).transpose(0, 2, 1, 3).reshape(G * Hkv, Sc, Dh),
                       Sp, theta, si).reshape(G, Hkv, Sc, Dh)
    Op, Os = pa.forward(qp, kp, vp, qs, ks, vs, si)        # resident, churn-free
    c.update(Op=Op, Os=Os)
    o_all = np.concatenate([Op.transpose(1, 0, 2).reshape(Sp, qd),
                            Os.transpose(0, 2, 1, 3).reshape(G * Sc, qd)], 0)
    c["o_tok"] = o_all
    xall = _bf(res + linear_bf16(o_all, w["o_proj"], si))

    res = xall
    h2, r2 = rmsnorm_forward(xall, w["post_ln"], si, cfg.eps)
    if ffn is None:                                        # dense SwiGLU
        gate = linear_bf16(h2, w["gate_proj"], si); up = linear_bf16(h2, w["up_proj"], si)
        act = swiglu_forward(gate, up, si)
        ff = linear_bf16(act, w["down_proj"], si)
        c.update(gate=gate, up=up, act=act)
    else:                                                  # routed MoE (GroupedMoEFFN, state on ffn)
        ff, _ = ffn.forward(h2, si)
    c.update(x2=xall, r2=r2, h2=h2)
    xall = _bf(res + ff)
    if return_cache:
        return xall[:Sp], xall[Sp:].reshape(G, Sc, H), c
    return xall[:Sp], xall[Sp:].reshape(G, Sc, H)


def prefix_layer_global(w, xp, xs, cfg, pa, si, ffn=None, return_cache=False):
    """GLOBAL (NoPE + full causal) prefix-shared layer — the MoE-model global case."""
    return prefix_layer(w, xp, xs, cfg, pa, si, ffn=ffn, theta=None, return_cache=return_cache)


def prefix_layer_local(w, xp, xs, cfg, pa, si, ffn=None, return_cache=False):
    """LOCAL (RoPE theta_local + sliding window) prefix-shared layer. pa must carry the window."""
    assert pa.window > 0, "local layer needs PrefixGlobalAttn(window=cfg.window)"
    return prefix_layer(w, xp, xs, cfg, pa, si, ffn=ffn, theta=cfg.rope_theta_local,
                        return_cache=return_cache)


def prefix_layer_bwd(w, d_xp, d_xs, c, cfg, pa, si, ffn=None, theta=None):
    """Reverse of prefix_layer. d_xp (Sp,H), d_xs (G,Sc,H) → (d_xp_in, d_xs_in, grads). Prompt KV
    grad = self + Σ_G cross (via PrefixGlobalAttn.backward). theta must match the forward's
    (None → NoPE, no rope-backward); ffn must match too (GroupedMoEFFN holds state from forward)."""
    from ancora.model.qwen3_layer import linear_bf16_backward
    from ancora.kernels.norm import rmsnorm_backward
    from ancora.kernels.activation import swiglu_backward
    from ancora.kernels.rope import rope_backward
    from ancora.rl.prefix_grpo import _rope_off_bwd
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    Sp, G, Sc = c["Sp"], c["G"], c["Sc"]; qd = Hq * Dh; M = Sp + G * Sc
    g = {}
    d = np.concatenate([d_xp.reshape(Sp, H), d_xs.reshape(G * Sc, H)], 0)

    d_res2 = d
    if ffn is None:                                        # dense SwiGLU backward
        d_a, g["down_proj"] = linear_bf16_backward(d, c["act"], w["down_proj"], si)
        d_gate, d_up = swiglu_backward(c["gate"], c["up"], d_a, si)
        d_h2a, g["gate_proj"] = linear_bf16_backward(d_gate, c["h2"], w["gate_proj"], si)
        d_h2b, g["up_proj"]   = linear_bf16_backward(d_up,   c["h2"], w["up_proj"],   si)
        d_h2 = d_h2a + d_h2b
    else:                                                  # routed MoE backward (state on ffn)
        d_h2, g["ffn"] = ffn.backward(d, stream_int=si)
    d_x2_mlp, g["post_ln"] = rmsnorm_backward(c["x2"], w["post_ln"], d_h2, c["r2"], si)
    d_x2 = d_res2 + d_x2_mlp

    d_res1 = d_x2
    d_o_tok, g["o_proj"] = linear_bf16_backward(d_x2, c["o_tok"], w["o_proj"], si)
    d_o_p = d_o_tok[:Sp].reshape(Sp, Hq, Dh).transpose(1, 0, 2)               # (Hq, Sp, Dh)
    d_o_s = d_o_tok[Sp:].reshape(G, Sc, Hq, Dh).transpose(0, 2, 1, 3)         # (G, Hq, Sc, Dh)
    dQp, dKp, dVp, dQs, dKs, dVs = pa.backward(c["Op"], c["Os"], d_o_p, d_o_s, si)  # resident
    if theta is not None:                                  # LOCAL: undo RoPE (prompt@0.., suffix@Sp..)
        dQp = rope_backward(dQp[None], si, theta)[0]; dKp = rope_backward(dKp[None], si, theta)[0]
        dQs = _rope_off_bwd(dQs.reshape(G * Hq,  Sc, Dh), Sp, theta, si).reshape(G, Hq,  Sc, Dh)
        dKs = _rope_off_bwd(dKs.reshape(G * Hkv, Sc, Dh), Sp, theta, si).reshape(G, Hkv, Sc, Dh)
    # head-major grads → token-major
    d_qn = np.concatenate([dQp.transpose(1, 0, 2).reshape(Sp, Hq, Dh),
                           dQs.transpose(0, 2, 1, 3).reshape(G * Sc, Hq, Dh)], 0).reshape(M * Hq, Dh)
    d_kn = np.concatenate([dKp.transpose(1, 0, 2).reshape(Sp, Hkv, Dh),
                           dKs.transpose(0, 2, 1, 3).reshape(G * Sc, Hkv, Dh)], 0).reshape(M * Hkv, Dh)
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


def prefix_layer_global_bwd(w, d_xp, d_xs, c, cfg, pa, si, ffn=None):
    """Backward of the GLOBAL (NoPE) prefix layer — no rope-backward."""
    return prefix_layer_bwd(w, d_xp, d_xs, c, cfg, pa, si, ffn=ffn, theta=None)


def prefix_layer_local_bwd(w, d_xp, d_xs, c, cfg, pa, si, ffn=None):
    """Backward of the LOCAL (RoPE + window) prefix layer."""
    return prefix_layer_bwd(w, d_xp, d_xs, c, cfg, pa, si, ffn=ffn, theta=cfg.rope_theta_local)


class PrefixGlobalAttn:
    """Churn-free prefix-shared attention. Persistent buffers sized for one group: prompt (Sp)
    shared + G completions (Sc each). Reuse across steps — never re-alloc.
    window=0 → GLOBAL (NoPE) full causal; window>0 → LOCAL sliding-window (caller ropes q/k —
    prompt at 0.., suffix at offset Sp.. — before forward; the windowed prefix kernels are
    bitwise == _attn_fwd_win on each [P, s_i], test_attn_prefix_win.py)."""
    def __init__(self, Hq, Hkv, D, Sp, Sc, G, window: int = 0):
        assert Sp % BKV == 0 and Sc % BQ == 0
        assert window % BKV == 0
        self.Hq, self.Hkv, self.D, self.Sp, self.Sc, self.G = Hq, Hkv, D, Sp, Sc, G
        self.window = window; self.WB = window // BKV          # 0 → global
        self.NQBp, self.NKVBp, self.NQBs, self.NKVBs = Sp // BQ, Sp // BKV, Sc // BQ, Sc // BKV
        self.scale = float(1.0 / math.sqrt(D))
        Z = _DBuf.zeros
        self.gQp = Z((Hq * Sp, D), np.uint16);  self.gKp = Z((Hkv * Sp, D), np.uint16); self.gVp = Z((Hkv * Sp, D), np.uint16)
        self.gQs = Z((G * Hq * Sc, D), np.uint16); self.gKs = Z((G * Hkv * Sc, D), np.uint16); self.gVs = Z((G * Hkv * Sc, D), np.uint16)
        self.gOp = Z((Hq * Sp, D), np.float32);   self.gLp = Z((Hq * Sp, 1), np.float32)
        self.gOs = Z((G * Hq * Sc, D), np.float32); self.gLs = Z((G * Hq * Sc, 1), np.float32)
        # backward grad buffers (persistent)
        self.GG = Hq // Hkv
        self.gdOp = Z((Hq * Sp, D), np.uint16); self.gdOs = Z((G * Hq * Sc, D), np.uint16)
        self.gDp = Z((Hq * Sp, 1), np.float32); self.gDs = Z((G * Hq * Sc, 1), np.float32)
        self.gdQp = Z((Hq * Sp, D), np.float32); self.gdKps = Z((Hkv * Sp, D), np.float32); self.gdVps = Z((Hkv * Sp, D), np.float32)
        self.gdKpc = Z((Hkv * Sp, D), np.float32); self.gdVpc = Z((Hkv * Sp, D), np.float32)
        self.gdQs = Z((G * Hq * Sc, D), np.float32); self.gdKs = Z((G * Hkv * Sc, D), np.float32); self.gdVs = Z((G * Hkv * Sc, D), np.float32)

    def forward(self, pq, pk, pv, sq, sk, sv, si):
        """pq/pk/pv: prompt (Hq|Hkv, Sp, D) f32; sq/sk/sv: suffix (G, Hq|Hkv, Sc, D) f32.
        → (Op (Hq,Sp,D), Os (G,Hq,Sc,D)). Uploads + 2 kernel launches on persistent buffers, no alloc."""
        Hq, Hkv, D, Sp, Sc, G = self.Hq, self.Hkv, self.D, self.Sp, self.Sc, self.G
        keep = [_up(self.gQp, _f32bf(pq.reshape(Hq * Sp, D)),  si),
                _up(self.gKp, _f32bf(pk.reshape(Hkv * Sp, D)), si),
                _up(self.gVp, _f32bf(pv.reshape(Hkv * Sp, D)), si),
                _up(self.gQs, _f32bf(sq.reshape(G * Hq * Sc, D)),  si),
                _up(self.gKs, _f32bf(sk.reshape(G * Hkv * Sc, D)), si),
                _up(self.gVs, _f32bf(sv.reshape(G * Hkv * Sc, D)), si)]
        if self.WB:                                            # LOCAL sliding-window
            ct.launch(si, (self.NQBp, Hq, 1), _attn_fwd_win,
                      (self.gQp, self.gKp, self.gVp, self.gOp, self.gLp, self.NQBp, self.NKVBp,
                       Hq, Hkv, self.scale, self.WB))
            ct.launch(si, (self.NQBs, G * Hq, 1), _attn_fwd_prefix_win,
                      (self.gQs, self.gKp, self.gVp, self.gKs, self.gVs, self.gOs, self.gLs,
                       self.NQBs, self.NKVBp, self.NKVBs, Hq, Hkv, self.scale, self.WB))
        else:                                                  # GLOBAL (NoPE) full causal
            ct.launch(si, (self.NQBp, Hq, 1), _attn_fwd,
                      (self.gQp, self.gKp, self.gVp, self.gOp, self.gLp, self.NQBp, self.NKVBp, Hq, Hkv, self.scale))
            ct.launch(si, (self.NQBs, G * Hq, 1), _attn_fwd_prefix,
                      (self.gQs, self.gKp, self.gVp, self.gKs, self.gVs, self.gOs, self.gLs,
                       self.NQBs, self.NKVBp, self.NKVBs, Hq, Hkv, self.scale))
        cudart.cudaStreamSynchronize(si)
        self._keep = keep
        return self.gOp.to_numpy().reshape(Hq, Sp, D), self.gOs.to_numpy().reshape(G, Hq, Sc, D)

    def backward(self, Op, Os, dOp, dOs, si):
        """dOp (Hq,Sp,D)=Σ_i dO_i[prefix] (shared prompt-output grad), dOs (G,Hq,Sc,D). Uses the
        forward's gKp/gVp/gQs/... + gLp/gLs (persistent). Returns (dQp,dKp,dVp, dQs,dKs,dVs). Prompt
        KV grad = self + Σ_G cross. All launches on persistent buffers (no alloc → churn-free)."""
        Hq, Hkv, D, Sp, Sc, G, GG = self.Hq, self.Hkv, self.D, self.Sp, self.Sc, self.G, self.GG
        sc = self.scale; NQBp, NKVBp, NQBs, NKVBs = self.NQBp, self.NKVBp, self.NQBs, self.NKVBs
        keep = [_up(self.gdOp, _f32bf(dOp.reshape(Hq * Sp, D)),  si),
                _up(self.gdOs, _f32bf(dOs.reshape(G * Hq * Sc, D)), si),
                _up(self.gDp, (Op * dOp).sum(-1).reshape(Hq * Sp, 1).astype(np.float32), si),
                _up(self.gDs, (Os * dOs).sum(-1).reshape(G * Hq * Sc, 1).astype(np.float32), si)]
        if self.WB:                                            # LOCAL sliding-window
            WB = self.WB
            ct.launch(si, (NQBs, G * Hq, 1), _attn_bwd_dq_prefix_win,
                      (self.gQs, self.gKp, self.gVp, self.gKs, self.gVs, self.gdOs, self.gLs, self.gDs,
                       self.gdQs, NQBs, NKVBp, NKVBs, Hq, Hkv, sc, WB))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix_win,
                      (self.gQs, self.gKp, self.gVp, self.gdOs, self.gLs, self.gDs, self.gdKpc, self.gdVpc,
                       NQBs, NKVBp, Hq, Hkv, G, GG, sc, WB))
            ct.launch(si, (NQBp, Hq, 1), _attn_bwd_dq_win,
                      (self.gQp, self.gKp, self.gVp, self.gdOp, self.gLp, self.gDp, self.gdQp,
                       NQBp, NKVBp, Hq, Hkv, sc, WB))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_win,
                      (self.gQp, self.gKp, self.gVp, self.gdOp, self.gLp, self.gDp, self.gdKps, self.gdVps,
                       NQBp, NKVBp, Hq, Hkv, GG, sc, WB))
            ct.launch(si, (NKVBs, G * Hkv, 1), _attn_bwd_dkdv_win,
                      (self.gQs, self.gKs, self.gVs, self.gdOs, self.gLs, self.gDs, self.gdKs, self.gdVs,
                       NQBs, NKVBs, Hq, Hkv, GG, sc, WB))
        else:                                                  # GLOBAL (NoPE) full causal
            # suffix dQ (prefix no-mask + suffix causal) + prompt CROSS dK/dV (Σ_G, no mask)
            ct.launch(si, (NQBs, G * Hq, 1), _attn_bwd_dq_prefix,
                      (self.gQs, self.gKp, self.gVp, self.gKs, self.gVs, self.gdOs, self.gLs, self.gDs, self.gdQs,
                       NQBs, NKVBp, NKVBs, Hq, Hkv, sc))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv_prefix,
                      (self.gQs, self.gKp, self.gVp, self.gdOs, self.gLs, self.gDs, self.gdKpc, self.gdVpc,
                       NQBs, NKVBp, Hq, Hkv, G, GG, sc))
            # prompt dQ + prompt SELF dK/dV (prompt self-attn)
            ct.launch(si, (NQBp, Hq, 1), _attn_bwd_dq,
                      (self.gQp, self.gKp, self.gVp, self.gdOp, self.gLp, self.gDp, self.gdQp, NQBp, NKVBp, Hq, Hkv, sc))
            ct.launch(si, (NKVBp, Hkv, 1), _attn_bwd_dkdv,
                      (self.gQp, self.gKp, self.gVp, self.gdOp, self.gLp, self.gDp, self.gdKps, self.gdVps,
                       NQBp, NKVBp, Hq, Hkv, GG, sc))
            # suffix dK/dV (suffix keys attended only by suffix queries, causal) — B=G
            ct.launch(si, (NKVBs, G * Hkv, 1), _attn_bwd_dkdv,
                      (self.gQs, self.gKs, self.gVs, self.gdOs, self.gLs, self.gDs, self.gdKs, self.gdVs,
                       NQBs, NKVBs, Hq, Hkv, GG, sc))
        cudart.cudaStreamSynchronize(si)
        self._bkeep = keep
        dKp = self.gdKps.to_numpy().reshape(Hkv, Sp, D) + self.gdKpc.to_numpy().reshape(Hkv, Sp, D)  # self + cross
        dVp = self.gdVps.to_numpy().reshape(Hkv, Sp, D) + self.gdVpc.to_numpy().reshape(Hkv, Sp, D)
        return (self.gdQp.to_numpy().reshape(Hq, Sp, D), dKp, dVp,
                self.gdQs.to_numpy().reshape(G, Hq, Sc, D),
                self.gdKs.to_numpy().reshape(G, Hkv, Sc, D), self.gdVs.to_numpy().reshape(G, Hkv, Sc, D))
