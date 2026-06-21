"""ResidentMoEDenseLayer (milestone 1 of the device-resident MoE training step): the resident
dense-FFN + local/global-attention layer vs the host MoEDecoderLayer(dense). Correctness (small
fp32-vs-bf16-residual gap) + the host-orchestration speedup. Foreground only (GPU-heavy)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, _bf
from ancora.model.resident_moe import ResidentMoEDenseLayer
from ancora.model.resident import _DBuf, _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])

_bits2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def rel(a, b): return np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / (np.abs(b).max() + 1e-9)
def sync(): cudart.cudaStreamSynchronize(si)


def _case(is_global, S):
    cfg = MoEConfig(); B = 2; M = B * S
    host = MoEDecoderLayer(cfg, is_global=is_global, ffn_dense=True, seed=3)
    weights = {**host.attn, **host.ffn.w}
    res = ResidentMoEDenseLayer(cfg, weights, B, S, is_global=is_global, window=cfg.window)
    rng = np.random.default_rng(0)
    x = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.5).astype(np.float32))
    dout = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.3).astype(np.float32))
    yh, ch = host.forward(x, si, return_cache=True); dxh, _ = host.backward(dout, ch, si)
    gx = _DBuf(x.reshape(M, cfg.hidden).astype(np.float32))        # fp32 residual stream
    gdo = _DBuf(_f32bf(dout.reshape(M, cfg.hidden)))               # grad-of-gout: bf16 bits
    res.forward(gx, si); sync(); yr = res.gout.to_numpy()
    res.backward(gdo, si); sync(); gdx = _bits2f(res.gdx.to_numpy())
    ef = rel(yr.reshape(B, S, -1), yh); ex = rel(gdx.reshape(B, S, -1), dxh)
    # timing
    yh, ch = host.forward(x, si, return_cache=True)
    t0 = time.perf_counter()
    for _ in range(5): yh, ch = host.forward(x, si, return_cache=True); host.backward(dout, ch, si)
    th = (time.perf_counter() - t0) / 5
    res.forward(gx, si); res.backward(gdo, si); sync()
    t0 = time.perf_counter()
    for _ in range(5): res.forward(gx, si); res.backward(gdo, si)
    sync(); tr = (time.perf_counter() - t0) / 5
    tag = "global+dense(NoPE)" if is_global else f"local+dense(window,S={S})"
    ok = ef < 0.02 and ex < 0.03                                  # fp32-vs-bf16-residual tolerance
    print(f"  {tag:26s} M={M:5d}: fwd {ef*100:.2f}% d_x {ex*100:.2f}%  | host {th*1e3:.0f}ms resident "
          f"{tr*1e3:.1f}ms = {th/tr:.0f}x  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ResidentMoEDenseLayer vs host MoEDecoderLayer(dense)")
    print("=" * 72)
    r = [_case(True, 128), _case(False, 1024)]
    print("=" * 72)
    print("  ALL PASS (resident correct + host-orchestration eliminated)" if all(r) else "  FAIL: " + str(r))
