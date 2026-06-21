"""ResidentMoELayer backward (milestone 2b): the grouped MoE-FFN backward resident + the shared
attention backward chain. Validated routing-consistently (vs-host-MoEDecoderLayer is meaningless for
MoE — resident routes on the fp32-residual h, host on bf16-residual h → boundary tokens flip).

Checks, on the SAME gh2 the resident forward routed on:
  (a) resident MoE-FFN backward (backward_resident) grads vs host GroupedMoEFFN.backward —
      d_h (grad-wrt-gh2), the 3 expert weight grads (device dWd/dWg/dWu), and the router grad.
  (b) the full layer backward runs end-to-end → a finite gdx (the attention chain is the already
      -validated _attn_bwd_chain shared with the dense layer).
  + the host-orchestration speedup of res.forward+backward vs host MoEDecoderLayer.forward+backward.
Foreground only (GPU-heavy)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoELayer
from ancora.kernels.moe import GroupedMoEFFN
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

_b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def rel(a, b): return np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9)
def sync(): cudart.cudaStreamSynchronize(si)


def _case(is_global, S):
    cfg = MoEConfig(); B = 2; M = B * S; E, Ie, H = 16, cfg.expert_inter, cfg.hidden
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=False, seed=4)
    res = ResidentMoELayer(cfg, host.attn, host.ffn.w, B, S, is_global=is_global, window=cfg.window)
    rng = np.random.default_rng(1)
    x = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.5).astype(np.float32))
    dout = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.3).astype(np.float32))
    gx = _DBuf(x.reshape(M, cfg.hidden).astype(np.float32))      # fp32 residual stream
    gdo = _DBuf(_f32bf(dout.reshape(M, cfg.hidden)))             # grad-of-gout: bf16 bits

    res.forward(gx, si); sync()
    gh2 = _b2f(res.gh2.to_numpy())                               # the FFN input the resident routed on
    res.backward(gdo, si); sync()
    d_h_r = _b2f(res.gdh2.to_numpy())                            # MoE-FFN grad-wrt-gh2 (expert+router)
    dWd_r = res.moe.dWd.to_numpy().reshape(E, Ie, H)
    dWg_r = res.moe.dWg.to_numpy().reshape(E, H, Ie)
    dWu_r = res.moe.dWu.to_numpy().reshape(E, H, Ie)
    Gr_r  = res.moe.G_router
    gdx   = _b2f(res.gdx.to_numpy())

    # reference: host GroupedMoEFFN backward on the SAME gh2 (→ same routing) + same dOut(=grad-of-gmlp)
    gm = GroupedMoEFFN(host.ffn.w, cfg.top_k, si)
    gm.forward(gh2)                                              # routes identically (same gh2)
    d_h_ref, g_ref = gm.backward(_b2f(gdo.to_numpy()))

    e_dh = rel(d_h_r, d_h_ref)
    e_wd = rel(dWd_r, g_ref["down_proj"]); e_wg = rel(dWg_r, g_ref["gate_proj"]); e_wu = rel(dWu_r, g_ref["up_proj"])
    e_rt = rel(Gr_r, g_ref["router"])
    fin  = np.isfinite(gdx).all() and np.abs(gdx).max() > 0

    # speedup vs host MoEDecoderLayer fwd+bwd
    yh, ch = host.forward(x, si, return_cache=True); host.backward(dout, ch, si)
    t0 = time.perf_counter()
    for _ in range(5): yh, ch = host.forward(x, si, return_cache=True); host.backward(dout, ch, si)
    th = (time.perf_counter() - t0) / 5
    res.forward(gx, si); res.backward(gdo, si); sync()
    t0 = time.perf_counter()
    for _ in range(5): res.forward(gx, si); res.backward(gdo, si)
    sync(); tr = (time.perf_counter() - t0) / 5

    tag = "global+MoE(NoPE)" if is_global else f"local+MoE(window,S={S})"
    ok = e_dh < 0.01 and max(e_wd, e_wg, e_wu) < 0.01 and e_rt < 0.01 and fin
    print(f"  {tag:24s} M={M:5d}: d_h {e_dh*100:.2f}%  dW(d/g/u) {e_wd*100:.2f}/{e_wg*100:.2f}/{e_wu*100:.2f}%  "
          f"router {e_rt*100:.3f}%  gdx={'finite' if fin else 'BAD'}  | host {th*1e3:.0f}ms resident {tr*1e3:.1f}ms "
          f"= {th/tr:.0f}x  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ResidentMoELayer backward (grouped MoE-FFN backward resident + shared attn chain)")
    print("=" * 90)
    r = [_case(True, 128), _case(False, 512)]
    print("=" * 90)
    print("  ALL PASS (resident MoE-FFN backward correct, routing-consistent)" if all(r) else "  FAIL: " + str(r))
