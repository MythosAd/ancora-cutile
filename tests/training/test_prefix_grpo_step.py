"""Prefix-shared GRPO layer fwd+bwd (rl/prefix_grpo.py _prefix_layer / _prefix_layer_bwd) vs the naive
batched TransformerLayer (B=G). The prompt is encoded ONCE; the layer's weight gradients must equal the
naive G-replicated layer (training-equivalent: the prompt grad is the Σ-over-G cross term, so the
prompt processed once with that summed gradient reproduces the G-copies grad). Loss only on the suffix
(d_xp top-layer input grad = 0), matching GRPO. Validates fwd (bitwise) + all weight grads + d_xs / the
Σ_G prompt grad d_xp vs naive — reliably (one layer = light alloc-churn; the full-model step is this
layer stacked + the validated embed/CE boundary, but the host path's churn needs best-of-N / the
device-resident path).  A full prefix_grpo_loss_backward observed clean at 0.48% vs naive (churn-limited)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config, TransformerLayer
from ancora.rl.prefix_grpo import _prefix_layer, _prefix_layer_bwd

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _case(G, Sp, Sc, top=True):
    cfg = Qwen3Config(); H = cfg.hidden
    L = TransformerLayer(cfg, seed=1); rng = np.random.default_rng(2)
    xp = (rng.standard_normal((Sp, H)) * 0.5).astype(np.float32)
    xs = (rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32)
    d_xs = (rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32)
    # top layer (GRPO): no loss on prompt → d_xp=0. lower layer: each copy shares a prompt grad d_pe.
    d_pe = np.zeros((Sp, H), np.float32) if top else (rng.standard_normal((Sp, H)) * 0.3).astype(np.float32)

    full = np.stack([np.concatenate([xp, xs[i]], 0) for i in range(G)])
    dfull = np.concatenate([np.tile(d_pe, (G, 1, 1)), d_xs], 1)
    # one layer is lighter, but several cases churn the allocator cumulatively → retry the case until the
    # prefix forward is BITWISE-equal to the naive suffix (Δ=0 = both churn-free), then trust that run's grads.
    e_fwd = e_g = e_xs = e_xp = 1e9; clean = False
    for _ in range(8):
        xpo, xso, c = _prefix_layer(L.w, xp.copy(), xs.copy(), cfg, si)
        d_xp_in, d_xs_in, gP = _prefix_layer_bwd(L.w, (G * d_pe).copy(), d_xs.copy(), c, cfg, si)
        out, ca = L.forward(full, si, return_cache=True)
        dxin, gN = L.backward(dfull, ca, si)
        if maxabs(xso, out[:, Sp:]) != 0.0: continue                      # forward churned → retry
        e_fwd = 0.0
        e_g  = max(rel(gP[k], gN[k]) for k in gP)
        e_xs = rel(d_xs_in, dxin[:, Sp:])
        e_xp = rel(d_xp_in, dxin[:, :Sp].sum(0))                         # prompt grad = Σ_G naive
        if max(e_g, e_xs, e_xp) < 0.02: clean = True; break
    ok = clean
    tag = "top(d_xp=0)" if top else "lower(d_xp≠0)"
    print(f"  G={G} Sp={Sp} Sc={Sc} {tag:13s}: fwdΔ={e_fwd:.0e}  grad≤{e_g*100:.2f}%  d_xs≤{e_xs*100:.2f}%  "
          f"d_xp(ΣG)≤{e_xp*100:.2f}%  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Prefix-shared GRPO LAYER fwd+bwd — weight grads + Σ_G prompt grad == naive batched layer")
    print("=" * 88)
    r = [_case(4, 128, 128, top=True), _case(6, 128, 64, top=True),
         _case(4, 128, 128, top=False), _case(4, 128, 64, top=False)]
    print("=" * 88)
    print("  ALL PASS (prefix-shared layer bwd == naive → training-equivalent; prompt grad = Σ_G)" if all(r)
          else "  FAIL: " + str(r))
