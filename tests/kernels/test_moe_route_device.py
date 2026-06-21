"""Device-resident MoE router (stage A gating + stage B dispatch) in GroupedMoEFFN.forward_resident.
The forward now does the router projection + softmax + top-k + the sort/dispatch ALL on device (no
host sync/download). Validates:
  (a) device gating (moe_router_gate, fp32) vs host _route — routing agreement (flips only on near
      ties) + top-k weight error,
  (b) device-routed forward vs host-routed forward on the SAME gh2 — equal where routing matches,
  (c) forward determinism (run twice → bitwise),
  (d) backward runs + weight grads finite.
Foreground only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEFFN
from ancora.kernels.moe import GroupedMoEFFN
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])
_b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def sync(): cudart.cudaStreamSynchronize(si)
def rel(a, b): return np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9)


def _case(M, seed=0):
    cfg = MoEConfig(); H, E, k = cfg.hidden, cfg.n_experts, cfg.top_k
    w = MoEFFN(cfg, np.random.default_rng(seed)).w
    rng = np.random.default_rng(seed + 1)
    h = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
    gh2 = _DBuf(_f32bf(h)); gmlp_d = _DBuf.zeros((M, H)); gmlp_h = _DBuf.zeros((M, H))

    dvc = GroupedMoEFFN(w, k, device_route=True)
    hst = GroupedMoEFFN(w, k, device_route=False)
    dvc.forward_resident(gh2, gmlp_d, si); sync()
    hst.forward_resident(gh2, gmlp_h, si); sync()

    # (a) device gating vs host _route
    hp, ht, hw = hst._route(_b2f(gh2.to_numpy()))
    dt = dvc.dtopi.to_numpy(); dw = dvc.dtopw.to_numpy()
    flip = int((np.sort(dt, 1) != np.sort(ht, 1)).any(1).sum())          # tokens whose expert SET differs
    same = ~(np.sort(dt, 1) != np.sort(ht, 1)).any(1)
    werr = rel(np.sort(dw[same], 1), np.sort(hw[same], 1)) if same.any() else 0.0

    # (b) device-routed vs host-routed forward (equal where routing matches)
    od, oh = _b2f(gmlp_d.to_numpy()), _b2f(gmlp_h.to_numpy())
    om = rel(od[same], oh[same]) if same.any() else 0.0                  # matched-token output error

    # (c) determinism
    dvc.forward_resident(gh2, gmlp_d, si); sync(); od2 = gmlp_d.to_numpy()
    dvc.forward_resident(gh2, gmlp_d, si); sync(); od3 = gmlp_d.to_numpy()
    det = np.abs(od2.astype(np.int64) - od3.astype(np.int64)).max()

    # (d) backward
    gdout = _DBuf(_f32bf((rng.standard_normal((M, H)) * 0.3).astype(np.float32)))
    gdh2 = _DBuf.zeros((M, H))
    dvc.backward_resident(gdout, gdh2, si); sync()
    fin = np.isfinite(_b2f(gdh2.to_numpy())).all() and np.isfinite(dvc.dWd.to_numpy()).all()

    flip_pct = 100.0 * flip / M
    ok = flip_pct < 2.0 and werr < 0.02 and om < 0.02 and det == 0 and fin
    print(f"  M={M:5d}: route-flip {flip}/{M} ({flip_pct:.2f}%)  topw {werr*100:.2f}%  matched-out {om*100:.2f}%  "
          f"det {'bitwise' if det == 0 else 'NO'}  bwd {'finite' if fin else 'BAD'}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("GroupedMoEFFN device-resident router (gate+dispatch on device, sync-free forward)")
    print("=" * 92)
    r = [_case(M, seed=s) for s, M in enumerate([256, 512, 1024])]
    print("=" * 92)
    print("  ALL PASS (device router: forward sync-free, routing ~matches host, deterministic)"
          if all(r) else "  FAIL: " + str(r))
