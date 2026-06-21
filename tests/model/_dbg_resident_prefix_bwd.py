"""Debug: locate the FIRST buffer where the resident-prefix suffix backward diverges from the
resident-replicated layer (global dense, Sp=128 Sc=128 G=4 — the Δbits=96 case)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoEDenseLayer
from ancora.model.resident_prefix import ResidentPrefixDenseLayer
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

Sp, Sc, G = 128, 128, 4
cfg = MoEConfig(window=128); H = cfg.hidden; S = Sp + Sc; M = Sp + G * Sc
host = MoEDecoderLayer(cfg, is_global=True, ffn_dense=True, seed=4)
w = {**host.attn, **host.ffn.w}
pre = ResidentPrefixDenseLayer(cfg, w, Sp, Sc, G, True, window=cfg.window)
ref = ResidentMoEDenseLayer(cfg, w, G, S, True, window=cfg.window)
Hq, Hkv, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

rng = np.random.default_rng(1)
xp = _bf((rng.standard_normal((Sp, H)) * 0.5).astype(np.float32))
xs = _bf((rng.standard_normal((G, Sc, H)) * 0.5).astype(np.float32))
dxs = _bf((rng.standard_normal((G, Sc, H)) * 0.3).astype(np.float32))
xpre = np.concatenate([xp, xs.reshape(G * Sc, H)], 0)
xrep = np.concatenate([np.concatenate([xp, xs[i]], 0) for i in range(G)], 0)
dpre = np.concatenate([np.zeros((Sp, H), np.float32), dxs.reshape(G * Sc, H)], 0)
drep = np.concatenate([np.concatenate([np.zeros((Sp, H), np.float32), dxs[i]], 0) for i in range(G)], 0)
gxp = _DBuf(xpre.astype(np.float32)); gdp = _DBuf(_f32bf(dpre))
gxr = _DBuf(xrep.astype(np.float32)); gdr = _DBuf(_f32bf(drep))

pre.forward(gxp, si); pre.backward(gdp, si); sync()
ref.forward(gxr, si); ref.backward(gdr, si); sync()


def cmp_tok(name, bp, br, width):
    """token-major (M_, width): prefix suffix rows Sp+i*Sc+p vs replicated i*S+Sp+p."""
    a = bp.to_numpy(); b = br.to_numpy()
    aS = a[Sp:].reshape(G, Sc, width)
    bS = b.reshape(G, S, width)[:, Sp:]
    d = np.abs(aS.astype(np.float64) - bS.astype(np.float64)).max()
    print(f"  {name:10s} tok  suffix Δ={d:.3e}")
    return d


def cmp_hm(name, bp, br, Hh):
    """head-major (M_*Hh, Dh): prefix row Hh*Sp + (i*Hh+h)*Sc+p vs replicated (i*Hh+h)*S + Sp+p."""
    a = bp.to_numpy(); b = br.to_numpy()
    aS = a[Hh * Sp:].reshape(G, Hh, Sc, -1)
    bS = b.reshape(G, Hh, S, -1)[:, :, Sp:]
    d = np.abs(aS.astype(np.float64) - bS.astype(np.float64)).max()
    print(f"  {name:10s} head suffix Δ={d:.3e}")
    return d


print("forward refs (sanity):")
cmp_tok("gx2", pre.gx2, ref.gx2, H)
print("backward chain (token-major then attention):")
cmp_tok("gdx2", pre.gdx2, ref.gdx2, H)
cmp_tok("gdotok", pre.gdotok, ref.gdotok, Hq * Dh)
cmp_hm("gdohm", pre.gdohm, ref.gdohm, Hq)
cmp_hm("gDelta", pre.gDelta, ref.gDelta, Hq)
cmp_hm("gdqr", pre.gdqr, ref.gdqr, Hq)
cmp_hm("gdkr", pre.gdkr, ref.gdkr, Hkv)
cmp_hm("gdvh", pre.gdvh, ref.gdvh, Hkv)
cmp_tok("gdqn", pre.gdqn, ref.gdqn, Hq * Dh)
cmp_tok("gdkn", pre.gdkn, ref.gdkn, Hkv * Dh)
cmp_tok("gdv", pre.gdv, ref.gdv, Hkv * Dh)
cmp_tok("gdq", pre.gdq, ref.gdq, Hq * Dh)
cmp_tok("gdh1", pre.gdh1, ref.gdh1, H)
cmp_tok("gdxa", pre.gdxa, ref.gdxa, H)
cmp_tok("gdx", pre.gdx, ref.gdx, H)
