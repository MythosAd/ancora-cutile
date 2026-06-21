"""Device-resident router BACKWARD (G_router weight-grad GEMM + d_h_router GEMM + gate-bwd) in
GroupedMoEFFN.backward_resident(device_route=True). With these, the MoE backward router grad is fully
on device — no gh2/topi/probs/dsg download. Validates vs the host router-backward path (device_route
=False) on the SAME weights/gh2/gdout (forward routing matches bitwise, test_moe_route_device.py):
  - G_router (H,E) device vs host,  d_h into gdh2 (M,H) device vs host,
  - expert weight grads dWd/dWg/dWu (should be ~bitwise — same kernels),
  - determinism (backward twice → bitwise gdh2 + G_router).
Foreground only."""
import sys, os
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


def _run(gm, gh2, gdout, M, H):
    gmlp = _DBuf.zeros((M, H)); gdh2 = _DBuf.zeros((M, H))
    gm.forward_resident(gh2, gmlp, si); sync()
    gm.backward_resident(gdout, gdh2, si); sync()
    return gdh2, gm


def _case(M, seed=0):
    cfg = MoEConfig(); H, E, k = cfg.hidden, cfg.n_experts, cfg.top_k
    w = MoEFFN(cfg, np.random.default_rng(seed)).w
    rng = np.random.default_rng(seed + 1)
    h = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
    gh2 = _DBuf(_f32bf(h))
    gdout = _DBuf(_f32bf((rng.standard_normal((M, H)) * 0.3).astype(np.float32)))

    dvc = GroupedMoEFFN(w, k, device_route=True)
    hst = GroupedMoEFFN(w, k, device_route=False)
    gdh2_d, dvc = _run(dvc, gh2, gdout, M, H)
    gdh2_h, hst = _run(hst, gh2, gdout, M, H)

    Gr_d = dvc.G_router_dev.to_numpy(); Gr_h = hst.G_router            # router weight grad
    e_gr = rel(Gr_d, Gr_h)
    e_dh = rel(_b2f(gdh2_d.to_numpy()), _b2f(gdh2_h.to_numpy()))       # grad wrt gh2 (expert + router)
    e_wd = rel(dvc.dWd.to_numpy(), hst.dWd.to_numpy())                 # expert weight grads
    e_wg = rel(dvc.dWg.to_numpy(), hst.dWg.to_numpy())
    e_wu = rel(dvc.dWu.to_numpy(), hst.dWu.to_numpy())

    # determinism: device backward twice → bitwise
    gdh2_2, _ = _run(dvc, gh2, gdout, M, H); a = gdh2_2.to_numpy().copy(); ga = dvc.G_router_dev.to_numpy().copy()
    gdh2_3, _ = _run(dvc, gh2, gdout, M, H); b = gdh2_3.to_numpy(); gb = dvc.G_router_dev.to_numpy()
    det = int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) + rel(ga, gb) if False else \
          max(int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()), float(np.abs(ga - gb).max()))

    ok = e_gr < 0.01 and e_dh < 0.02 and max(e_wd, e_wg, e_wu) < 0.01 and det == 0
    print(f"  M={M:5d}: G_router {e_gr*100:.3f}%  d_h(gdh2) {e_dh*100:.2f}%  dW(d/g/u) "
          f"{e_wd*100:.2f}/{e_wg*100:.2f}/{e_wu*100:.2f}%  det {'bitwise' if det == 0 else det}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("device-resident router BACKWARD (G_router + d_h GEMMs on device) vs host path")
    print("=" * 90)
    r = [_case(M, seed=s) for s, M in enumerate([256, 512, 1024])]
    print("=" * 90)
    print("  ALL PASS (router backward fully on device, matches host, deterministic)" if all(r)
          else "  FAIL: " + str(r))
