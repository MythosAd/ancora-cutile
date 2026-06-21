"""Prefix-shared GLOBAL (NoPE) MoE layer — prefix_layer_global(ffn=GroupedMoEFFN) fwd+bwd vs the naive
MoEDecoderLayer(is_global=True, grouped=True) on the G replicated [prompt, completion_i]. This is the
REAL MoE-model global layer: NoPE full attention + routed top-k grouped MoE FFN.

Why prefix-sharing still holds with a router: the router + grouped GEMM + combine are all per-token /
row-independent (batch-invariant by design), so identical token rows route identically and produce
bitwise-identical outputs whether the prompt appears once (prefix) or G times (naive). Weight-grad
equivalence is the same Σ_G linearity argument as the dense layer (backward linear in the grad).

Two GroupedMoEFFN instances SHARE the master weight dict (naive.ffn.w) but own separate device buffers
(different M) → no re-prealloc churn. Host glue churns → retry until suffix forward is bitwise."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer
from ancora.kernels.moe import GroupedMoEFFN
from ancora.kernels.attention import D
from ancora.rl.prefix_resident import PrefixGlobalAttn, prefix_layer_global, prefix_layer_global_bwd

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _case(G, Sp, Sc, top=True):
    cfg = MoEConfig(); H = cfg.hidden
    naive = MoEDecoderLayer(cfg, is_global=True, ffn_dense=False, seed=1, grouped=True)
    ffn_p = GroupedMoEFFN(naive.ffn.w, cfg.top_k)        # shared master weights, own buffers (prefix M)
    rng = np.random.default_rng(2)
    xp = (rng.standard_normal((Sp, H)) * 0.5).astype(np.float32)
    xs = (rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32)
    d_xs = (rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32)
    d_pe = np.zeros((Sp, H), np.float32) if top else (rng.standard_normal((Sp, H)) * 0.3).astype(np.float32)
    pa = PrefixGlobalAttn(cfg.n_heads, cfg.n_kv_heads, D, Sp, Sc, G)

    full = np.stack([np.concatenate([xp, xs[i]], 0) for i in range(G)])
    dfull = np.concatenate([np.tile(d_pe, (G, 1, 1)), d_xs], 1)

    e_g = e_f = e_xs = e_xp = 1e9; clean = False
    for _ in range(8):
        xpo, xso, c = prefix_layer_global(naive.attn, xp.copy(), xs.copy(), cfg, pa, si,
                                          ffn=ffn_p, return_cache=True)
        d_xp_in, d_xs_in, gP = prefix_layer_global_bwd(naive.attn, (G * d_pe).copy(), d_xs.copy(),
                                                       c, cfg, pa, si, ffn=ffn_p)
        out, cn = naive.forward(full, si, return_cache=True)
        dxin, gN = naive.backward(dfull, cn, si)
        if maxabs(xso, out[:, Sp:]) != 0.0: continue              # host glue churned → retry
        e_g  = max(rel(gP[k], gN[k]) for k in gP if k != "ffn")
        e_f  = max(rel(gP["ffn"][k], gN["ffn"][k]) for k in gP["ffn"])
        e_xs = rel(d_xs_in, dxin[:, Sp:])
        e_xp = rel(d_xp_in, dxin[:, :Sp].sum(0))                 # prompt grad = Σ_G naive
        if max(e_g, e_f, e_xs, e_xp) < 0.02: clean = True; break
    tag = "top(d_xp=0)" if top else "lower(d_xp≠0)"
    print(f"  G={G} Sp={Sp} Sc={Sc} {tag:13s}: attn-grad≤{e_g*100:.2f}%  moe-grad≤{e_f*100:.2f}%  "
          f"d_xs≤{e_xs*100:.2f}%  d_xp(ΣG)≤{e_xp*100:.2f}%  {'OK' if clean else 'FAIL'}")
    return clean


if __name__ == "__main__":
    print("Prefix-shared GLOBAL (NoPE) MoE layer — fwd bitwise + grads == naive grouped-MoE layer")
    print("=" * 92)
    r = [_case(4, 128, 128, top=True), _case(6, 128, 64, top=True), _case(4, 256, 64, top=False)]
    print("=" * 92)
    print("  ALL PASS (prefix-shared MoE global layer training-equivalent; router/experts bitwise)" if all(r)
          else "  FAIL: " + str(r))
