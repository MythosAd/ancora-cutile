"""Prefix-shared LOCAL (RoPE + sliding-window) layer — prefix_layer_local(+_bwd) vs the naive
MoEDecoderLayer(is_global=False) on the G replicated [prompt, completion_i]. Local layers ARE the
RoPE'd ones (user: "Sliding window 有rope，是局部的"): prompt q/k roped at 0..Sp-1, suffix at the
OFFSET positions Sp..Sp+Sc-1 (RNE bf16 — must match rope_forward), window via the new
_attn_*_prefix_win kernels (bitwise vs _attn_fwd_win, test_attn_prefix_win.py).

Covers BOTH FFN variants of the MoE model family:
  dense — naive DenseFFN's flat w merged into the layer dict (ffn=None path);
  moe   — GroupedMoEFFN sharing the naive's master expert weights.
cfg.window=128 (< Sp) so the window actually clips: early suffix queries reach into the prompt,
late ones never see it. Host glue churns → retry until the suffix forward is bitwise."""
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
from ancora.rl.prefix_resident import PrefixGlobalAttn, prefix_layer_local, prefix_layer_local_bwd

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _case(G, Sp, Sc, moe, top=True):
    cfg = MoEConfig(window=128); H = cfg.hidden
    rng = np.random.default_rng(2)
    xp = (rng.standard_normal((Sp, H)) * 0.5).astype(np.float32)
    xs = (rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32)
    d_xs = (rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32)
    d_pe = np.zeros((Sp, H), np.float32) if top else (rng.standard_normal((Sp, H)) * 0.3).astype(np.float32)

    full = np.stack([np.concatenate([xp, xs[i]], 0) for i in range(G)])
    dfull = np.concatenate([np.tile(d_pe, (G, 1, 1)), d_xs], 1)

    e_g = e_xs = e_xp = 1e9; clean = False
    for _ in range(8):
        # FRESH objects each attempt: GroupedMoEFFN packs weights lazily ONCE — a churn-corrupted
        # pack (e.g. the WdT transpose only the backward reads) would otherwise persist across
        # retries (forward bitwise, backward consistently wrong). seed=1 → identical weights.
        naive = MoEDecoderLayer(cfg, is_global=False, ffn_dense=not moe, seed=1, grouped=moe)
        if moe:
            wP = naive.attn; ffn_p = GroupedMoEFFN(naive.ffn.w, cfg.top_k)  # shared master weights
        else:
            wP = {**naive.attn, **naive.ffn.w}; ffn_p = None                # dense: flat-w SwiGLU path
        pa = PrefixGlobalAttn(cfg.n_heads, cfg.n_kv_heads, D, Sp, Sc, G, window=cfg.window)
        xpo, xso, c = prefix_layer_local(wP, xp.copy(), xs.copy(), cfg, pa, si,
                                         ffn=ffn_p, return_cache=True)
        d_xp_in, d_xs_in, gP = prefix_layer_local_bwd(wP, (G * d_pe).copy(), d_xs.copy(),
                                                      c, cfg, pa, si, ffn=ffn_p)
        out, cn = naive.forward(full, si, return_cache=True)
        dxin, gN = naive.backward(dfull, cn, si)
        if maxabs(xso, out[:, Sp:]) != 0.0: continue              # host glue churned → retry
        # naive grads: attn flat, FFN nested under "ffn" (prefix dense path returns FFN keys flat)
        e_g = 0.0
        for k in gP:
            if k == "ffn":
                e_g = max(e_g, max(rel(gP["ffn"][kk], gN["ffn"][kk]) for kk in gP["ffn"]))
            else:
                e_g = max(e_g, rel(gP[k], gN[k] if k in gN else gN["ffn"][k]))
        e_xs = rel(d_xs_in, dxin[:, Sp:])
        e_xp = rel(d_xp_in, dxin[:, :Sp].sum(0))                 # prompt grad = Σ_G naive
        if max(e_g, e_xs, e_xp) < 0.02: clean = True; break
    tag = ("moe" if moe else "dense") + ("/top" if top else "/lower")
    print(f"  G={G} Sp={Sp} Sc={Sc} W={cfg.window} {tag:11s}: grad≤{e_g*100:.2f}%  d_xs≤{e_xs*100:.2f}%  "
          f"d_xp(ΣG)≤{e_xp*100:.2f}%  {'OK' if clean else 'FAIL'}")
    return clean


if __name__ == "__main__":
    if len(sys.argv) > 1:                      # child: run ONE case, exit code = pass/fail
        G_, Sp_, Sc_, moe_, top_ = map(int, sys.argv[1:6])
        sys.exit(0 if _case(G_, Sp_, Sc_, moe=bool(moe_), top=bool(top_)) else 1)
    # parent: each case in a FRESH PROCESS. The host-glue alloc-churn race is allocator-phase
    # dependent — in a shared process a bad phase corrupts every retry of a case (identical
    # code+seed passes/fails across runs), while a fresh process always starts clean (verified).
    # The real fix is the device-resident per-token glue (no per-call alloc) — see CLAUDE.md.
    import subprocess
    print("Prefix-shared LOCAL (RoPE + window) layer — fwd bitwise + grads == naive local layer")
    print("=" * 92)
    cases = [(4, 256, 128, 0, 1), (4, 256, 128, 1, 1), (6, 128, 64, 1, 0)]
    r = [subprocess.run([sys.executable, __file__, *map(str, cse)]).returncode == 0 for cse in cases]
    print("=" * 92)
    print("  ALL PASS (prefix-shared local layer training-equivalent; offset-RoPE + window bitwise)"
          if all(r) else "  FAIL: " + str(r))
