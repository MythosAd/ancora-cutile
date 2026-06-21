"""ResidentMoELayer closed-loop training step (milestone 3): forward → backward → device-AdamW,
all device-resident. Overfits a single MoE layer to a fixed (input, target) pair via MSE and watches
the loss collapse — proving the device AdamW over the 3D expert weights (Wg/Wu/Wd updated in place +
transposed-weight refresh) + the host router AdamW actually close the loop. A frozen control (no step)
must stay flat. Foreground only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart, driver as cdrv

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoELayer
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

_b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def sync(): cudart.cudaStreamSynchronize(si)


def _overfit(is_global, S, steps=40, lr=3e-3, do_step=True):
    cfg = MoEConfig(); B = 2; M = B * S; H = cfg.hidden
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=False, seed=7)
    res = ResidentMoELayer(cfg, host.attn, host.ffn.w, B, S, is_global=is_global, window=cfg.window)
    rng = np.random.default_rng(2)
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    gx = _DBuf(x.reshape(M, H).astype(np.float32))               # fixed fp32-residual input
    target = (rng.standard_normal((M, H)) * 0.3).astype(np.float32)
    gd = _DBuf.zeros((M, H), np.uint16)                          # persistent grad buffer (no per-step alloc)
    L0 = None
    for it in range(steps):
        out = res.forward(gx, si); sync()
        o = out.to_numpy()
        diff = o - target
        L = float((diff * diff).mean())
        if it == 0: L0 = L
        g = (2.0 / o.size) * diff                                # MSE grad wrt gout
        h = np.ascontiguousarray(_f32bf(g))
        cdrv.cuMemcpyHtoDAsync(gd.ptr, h, h.nbytes, si)          # upload on si (same stream as backward)
        res.backward(gd, si)
        if do_step: res.step(si, lr)
        sync()
        if it in (0, steps // 2, steps - 1):
            print(f"      step {it:3d}: MSE {L:.6e}")
    Lf = L
    return L0, Lf


def _case(is_global, S):
    tag = "global+MoE(NoPE)" if is_global else f"local+MoE(window,S={S})"
    print(f"  {tag}  M={2*S}:")
    L0, Lf = _overfit(is_global, S, do_step=True)
    Lc0, Lcf = _overfit(is_global, S, steps=8, do_step=False)
    drop = Lf / L0; ctrl = Lcf / Lc0
    ok = drop < 0.25 and ctrl > 0.9                              # train collapses; frozen control flat
    print(f"    train {L0:.4e}->{Lf:.4e} ({drop*100:.1f}%)  frozen-control {ctrl*100:.1f}%  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ResidentMoELayer closed-loop training step (fwd->bwd->device-AdamW)")
    print("=" * 78)
    r = [_case(True, 128), _case(False, 512)]
    print("=" * 78)
    print("  ALL PASS (resident MoE training step closes the loop)" if all(r) else "  FAIL: " + str(r))
