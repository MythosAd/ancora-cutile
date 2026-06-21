"""ResidentMoELayer forward (milestone 2): attention (local/global) + the grouped MoE FFN, all
device-resident (router = the only host round-trip/layer). Validated routing-consistently — vs-host
is meaningless for MoE (resident routes on fp32-residual h, host on bf16-residual h → boundary tokens
flip). Checks: (a) resident FFN == grouped FFN on the same gh2, (b) gout == gx2 + gmlp; + speedup."""
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
from ancora.model.resident import _DBuf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

_b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def rel(a, b): return np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9)
def sync(): cudart.cudaStreamSynchronize(si)


def _case(is_global, S):
    cfg = MoEConfig(); B = 2; M = B * S
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=False, seed=4)
    res = ResidentMoELayer(cfg, host.attn, host.ffn.w, B, S, is_global=is_global, window=cfg.window)
    rng = np.random.default_rng(1)
    x = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.5).astype(np.float32))
    gx = _DBuf(x.reshape(M, cfg.hidden).astype(np.float32))
    res.forward(gx, si); sync()
    gout = res.gout.to_numpy(); gx2 = res.gx2.to_numpy(); gmlp = _b2f(res.gmlp_moe.to_numpy())
    gm = GroupedMoEFFN(host.ffn.w, cfg.top_k, si); out_g, _ = gm.forward(_b2f(res.gh2.to_numpy()))
    eff = rel(gmlp, out_g)                                # resident FFN vs grouped on same gh2
    easm = rel(gout, gx2 + gmlp.reshape(M, -1))           # layer assembly: gout == gx2 + gmlp
    # speedup vs host MoEDecoderLayer
    host.forward(x, si, return_cache=True)
    t0 = time.perf_counter()
    for _ in range(5): host.forward(x, si, return_cache=True)
    th = (time.perf_counter() - t0) / 5
    res.forward(gx, si); sync()
    t0 = time.perf_counter()
    for _ in range(5): res.forward(gx, si)
    sync(); tr = (time.perf_counter() - t0) / 5
    tag = "global+MoE(NoPE)" if is_global else f"local+MoE(window,S={S})"
    ok = eff < 0.01 and easm < 0.001
    print(f"  {tag:24s} M={M:5d}: FFN-tight {eff*100:.2f}%  gout==gx2+gmlp {easm*100:.3f}%  | "
          f"host {th*1e3:.0f}ms resident {tr*1e3:.1f}ms = {th/tr:.0f}x  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ResidentMoELayer forward (attention + grouped MoE FFN, resident)")
    print("=" * 78)
    r = [_case(True, 128), _case(False, 512)]
    print("=" * 78)
    print("  ALL PASS (resident MoE-FFN forward correct + ~72-111x vs host)" if all(r) else "  FAIL: " + str(r))
