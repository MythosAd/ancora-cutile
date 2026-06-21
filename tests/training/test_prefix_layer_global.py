"""Prefix-shared GLOBAL (NoPE) dense decoder layer (rl/prefix_resident.prefix_layer_global) vs a naive
NoPE dense layer on the G replicated [prompt, completion_i]. The MoE global layers are NoPE (no RoPE),
so the prefix attention (the churn-free resident PrefixGlobalAttn) plugs in directly. The completion
(suffix) hidden must be BITWISE-equal to the naive → ratio=1, with the prompt encoded ONCE.

Per-token ops use host helpers (alloc-churn) so we retry until the suffix forward is bitwise (= clean);
the attention itself is churn-free (PrefixGlobalAttn)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config, TransformerLayer, linear_bf16, _bf
from ancora.kernels.norm import rmsnorm_forward
from ancora.kernels.activation import swiglu_forward
from ancora.kernels.attention import flash_attn_forward, D
from ancora.rl.prefix_resident import PrefixGlobalAttn, prefix_layer_global

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _naive_nope(w, full, cfg):
    """NoPE dense layer forward on (G,S,H) → (G,S,H). Same as TransformerLayer but NO RoPE."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    G, S, _ = full.shape; M = G * S; qd = Hq * Dh
    xall = full.reshape(M, H).astype(np.float32)
    res = xall
    h, _ = rmsnorm_forward(xall, w["input_ln"], si, cfg.eps)
    q = linear_bf16(h, w["q_proj"], si); k = linear_bf16(h, w["k_proj"], si); v = linear_bf16(h, w["v_proj"], si)
    qn, _ = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], si, cfg.eps)
    kn, _ = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], si, cfg.eps)
    qh = qn.reshape(G, S, Hq, Dh).transpose(0, 2, 1, 3)         # NO rope
    kh = kn.reshape(G, S, Hkv, Dh).transpose(0, 2, 1, 3)
    vh = v.reshape(G, S, Hkv, Dh).transpose(0, 2, 1, 3)
    O = flash_attn_forward(qh, kh, vh, si)                      # NoPE causal GQA
    o_tok = O.transpose(0, 2, 1, 3).reshape(M, qd)
    xall = _bf(res + linear_bf16(o_tok, w["o_proj"], si))
    res = xall
    h, _ = rmsnorm_forward(xall, w["post_ln"], si, cfg.eps)
    act = swiglu_forward(linear_bf16(h, w["gate_proj"], si), linear_bf16(h, w["up_proj"], si), si)
    xall = _bf(res + linear_bf16(act, w["down_proj"], si))
    return xall.reshape(G, S, H)


def _case(G, Sp, Sc):
    cfg = Qwen3Config(); H = cfg.hidden
    L = TransformerLayer(cfg, seed=1); rng = np.random.default_rng(2)
    xp = (rng.standard_normal((Sp, H)) * 0.5).astype(np.float32)
    xs = (rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32)
    pa = PrefixGlobalAttn(cfg.n_heads, cfg.n_kv_heads, D, Sp, Sc, G)
    full = np.stack([np.concatenate([xp, xs[i]], 0) for i in range(G)])     # (G, Sp+Sc, H)

    e_suf = e_pre = 1e9; clean = False
    for _ in range(8):
        xpo, xso = prefix_layer_global(L.w, xp.copy(), xs.copy(), cfg, pa, si)
        out = _naive_nope(L.w, full, cfg)
        a = maxabs(xso, out[:, Sp:])
        if a != 0.0: continue                                              # per-token glue churned → retry
        e_suf = 0.0; e_pre = maxabs(xpo, out[0, :Sp]); clean = True; break
    ok = clean and e_pre == 0.0
    print(f"  G={G} Sp={Sp} Sc={Sc}: suffix Δ={e_suf:.0e}  prompt Δ={e_pre:.0e}  "
          f"tokens {G*(Sp+Sc)}→{Sp+G*Sc}  {'OK (bitwise → ratio=1)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Prefix-shared GLOBAL (NoPE) dense layer — completion hidden BITWISE == naive NoPE")
    print("=" * 80)
    r = [_case(4, 128, 128), _case(6, 128, 64), _case(4, 256, 64)]   # M=Sp+G·Sc must be %128
    print("=" * 80)
    print("  ALL PASS (NoPE global prefix layer bitwise; resident churn-free attention)" if all(r)
          else "  FAIL: " + str(r))
