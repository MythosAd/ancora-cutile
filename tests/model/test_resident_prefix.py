"""Device-resident prefix-shared layer (model/resident_prefix.py) vs the resident REPLICATED layer
(ResidentMoEDense/MoELayer, B=G on [prompt, suffix_i]) — both fully device-resident on persistent
buffers, so the comparison is churn-free and bitwise is meaningful (unlike the host-glue layers).

  (a) FORWARD: suffix rows AND prompt rows BITWISE == replicated (per-token ops are row-independent,
      prefix attention kernels bitwise == full attention) → ratio=1 with the prompt encoded ONCE
      (ratio=1 needs the FORWARD logprobs; gradients need correctness + determinism, not bitwise).
  (b) BACKWARD: suffix input-grad ULP-equal to replicated (≤0.5% — _attn_bwd_dq_prefix and
      _attn_bwd_dq have character-identical bodies but are separately compiled, and the tile
      compiler's FMA contraction differs → ~1-ULP f32 in dQ, amplified by the bf16 casts;
      verified via _dbg_resident_prefix_bwd.py: first divergence is gdqr at 8.6e-06 = 1 ULP of
      O(100), dK/dV exact); prompt input-grad == Σ_G (≤2%); weight grads ≤1% (row-order).
  (c) DETERMINISM: repeated fwd+bwd bitwise-identical (the alloc-churn that plagued the host-glue
      prefix tests is gone — this layer is the fix).
Covers dense GLOBAL (NoPE), dense LOCAL (RoPE+window), and grouped-MoE GLOBAL. Foreground only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoEDenseLayer, ResidentMoELayer
from ancora.model.resident_prefix import ResidentPrefixDenseLayer, ResidentPrefixMoELayer
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
_b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def rel(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9))
def sync(): cudart.cudaStreamSynchronize(si)


def _case(is_global, moe, Sp, Sc, G):
    cfg = MoEConfig(window=128); H = cfg.hidden; S = Sp + Sc; M = Sp + G * Sc
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=not moe, seed=4, grouped=moe)
    if moe:
        pre = ResidentPrefixMoELayer(cfg, host.attn, host.ffn.w, Sp, Sc, G, is_global, window=cfg.window)
        ref = ResidentMoELayer(cfg, host.attn, host.ffn.w, G, S, is_global, window=cfg.window)
    else:
        w = {**host.attn, **host.ffn.w}
        pre = ResidentPrefixDenseLayer(cfg, w, Sp, Sc, G, is_global, window=cfg.window)
        ref = ResidentMoEDenseLayer(cfg, w, G, S, is_global, window=cfg.window)

    rng = np.random.default_rng(1)
    xp = _bf((rng.standard_normal((Sp, H)) * 0.5).astype(np.float32))
    xs = _bf((rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32))
    dxs = _bf((rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32))   # top GRPO: prompt grad = 0
    xpre = np.concatenate([xp, xs.reshape(G * Sc, H)], 0)
    xrep = np.concatenate([np.concatenate([xp, xs[i]], 0) for i in range(G)], 0)
    dpre = np.concatenate([np.zeros((Sp, H), np.float32), dxs.reshape(G * Sc, H)], 0)
    drep = np.concatenate([np.concatenate([np.zeros((Sp, H), np.float32), dxs[i]], 0) for i in range(G)], 0)
    gxp = _DBuf(xpre.astype(np.float32)); gdp = _DBuf(_f32bf(dpre))
    gxr = _DBuf(xrep.astype(np.float32)); gdr = _DBuf(_f32bf(drep))

    pre.forward(gxp, si); pre.backward(gdp, si); sync()
    outP = pre.gout.to_numpy(); gdxP = pre.gdx.to_numpy()
    GP = {n: pre.G[n].to_numpy() for n in pre.PROJ + pre.NORM}
    ref.forward(gxr, si); ref.backward(gdr, si); sync()
    outR = ref.gout.to_numpy().reshape(G, S, H); gdxR = ref.gdx.to_numpy().reshape(G, S, H)
    GR = {n: ref.G[n].to_numpy() for n in ref.PROJ + ref.NORM}

    # (a) forward: suffix + prompt bitwise (f32 gout)
    e_fs = float(np.abs(outP[Sp:].reshape(G, Sc, H) - outR[:, Sp:]).max())
    e_fp = float(np.abs(outP[:Sp] - outR[0, :Sp]).max())
    # (b) backward: suffix input-grad ULP-equal (compiler FMA contraction, see header); prompt = Σ_G
    e_bs = rel(_b2f(gdxP[Sp:]).reshape(G, Sc, H), _b2f(gdxR[:, Sp:].reshape(G * Sc, H)).reshape(G, Sc, H))
    e_bp = rel(_b2f(gdxP[:Sp]), _b2f(gdxR[:, :Sp].reshape(G * Sp, H)).reshape(G, Sp, H).sum(0))
    e_g = max(rel(GP[n], GR[n]) for n in GP if GP[n].any() or GR[n].any())
    if moe:   # expert + router grads (both route on bitwise-equal h2 rows → same experts)
        e_g = max(e_g, rel(pre.moe.dWd.to_numpy(), ref.moe.dWd.to_numpy()),
                  rel(pre.moe.dWg.to_numpy(), ref.moe.dWg.to_numpy()),
                  rel(pre.moe.dWu.to_numpy(), ref.moe.dWu.to_numpy()),
                  rel(pre.moe.G_router, ref.moe.G_router))
    # (c) determinism: repeat fwd+bwd, everything bitwise
    pre.forward(gxp, si); pre.backward(gdp, si); sync()
    det = max(float(np.abs(pre.gout.to_numpy() - outP).max()),
              float(np.abs(pre.gdx.to_numpy().astype(np.int64) - gdxP.astype(np.int64)).max()),
              float(np.abs(pre.G["q_proj"].to_numpy() - GP["q_proj"]).max()))

    ok = e_fs == 0.0 and e_fp == 0.0 and e_bs < 0.005 and e_bp < 0.02 and e_g < 0.01 and det == 0.0
    tag = ("global" if is_global else "local ") + ("+MoE  " if moe else "+dense")
    print(f"  {tag} G={G} Sp={Sp} Sc={Sc}: fwd suf/pro Δ={e_fs:.0e}/{e_fp:.0e}  bwd suf≤{e_bs*100:.3f}%  "
          f"pro(ΣG)≤{e_bp*100:.2f}%  grads≤{e_g*100:.2f}%  det Δ={det:.0e}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Device-resident PREFIX-SHARED layer vs resident replicated layer — bitwise + churn-free")
    print("=" * 100)
    r = [_case(True,  False, 128, 128, 4),    # dense GLOBAL (NoPE)
         _case(False, False, 256, 64, 4),     # dense LOCAL (RoPE + window=128, prompt > window)
         _case(True,  True,  128, 128, 4)]    # grouped-MoE GLOBAL
    print("=" * 100)
    print("  ALL PASS (resident prefix layer bitwise == replicated → ratio=1, prompt encoded once, no churn)"
          if all(r) else "  FAIL: " + str(r))
