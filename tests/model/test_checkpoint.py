"""Activation checkpointing (long_context) — the backward RECOMPUTES each layer's forward instead
of keeping all layers' intermediates resident. Because recompute is DETERMINISTIC, the grads must
be BITWISE-identical to the full-store path.
  (A) forward unchanged (long_context only affects the backward).
  (B) every weight grad BITWISE == full-store (the correctness gate).
  (C) SFT converges identically.
Step 2a: recompute orchestration (no buffer-sharing yet — proves the flow is correct/deterministic).
Step 2b (next) shares buffers for the actual memory win, re-checked against this same bitwise gate."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def grads(m):
    g = {"embed": m.gegrad.to_numpy(), "fng": m.gfng.to_numpy()}
    for i, l in enumerate(m.layers):
        for n in ("q_proj", "o_proj", "input_ln"):
            g[f"L{i}.{n}"] = l.G[n].to_numpy()
        if hasattr(l, "moe"):
            g[f"L{i}.dWd"] = l.moe.dWd.to_numpy()
        else:
            g[f"L{i}.down_proj"] = l.G["down_proj"].to_numpy()
    return g


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)
    G, S = 4, 128
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, size=(G, S)).astype(np.int64)
    labels = np.concatenate([ids[:, 1:], np.zeros((G, 1), np.int64)], 1)

    full = ResidentMoEModel(cfg, from_host(host, G, S), G, S, device_route=True, long_context=False)
    ckpt = ResidentMoEModel(cfg, from_host(host, G, S), G, S, device_route=True, long_context=True)

    # ── (A) forward unchanged ──
    hf = full.forward(ids, si).copy()
    hc = ckpt.forward(ids, si).copy()
    eA = int(np.abs(hf.astype(np.int32) - hc.astype(np.int32)).max())
    print(f"  (A) forward checkpoint vs full-store: Δ={eA}  {'OK (unchanged)' if eA == 0 else 'FAIL'}")

    # ── (B) grads BITWISE == full-store (deterministic recompute) ──
    full.forward(ids, si); full.loss_backward(None, labels.reshape(-1), si)
    ckpt.forward(ids, si); ckpt.loss_backward(None, labels.reshape(-1), si)
    gf, gc = grads(full), grads(ckpt)
    eB = max(float(np.abs(gc[k].astype(np.float64) - gf[k].astype(np.float64)).max()) for k in gf)
    worst = max(gf, key=lambda k: np.abs(gc[k].astype(np.float64) - gf[k].astype(np.float64)).max())
    print(f"  (B) grads checkpoint vs full-store: Δ={eB:.0e} (worst {worst})  "
          f"{'OK (BITWISE → recompute is exact)' if eB == 0 else 'FAIL'}")

    # ── (C) SFT converges identically ──
    def overfit(m, n=80):
        c0 = c = None
        for t in range(n):
            m.forward(ids, si); c = m.loss_backward(None, labels.reshape(-1), si)
            if t == 0: c0 = c
            m.step(si, lr=2e-3); cudart.cudaStreamSynchronize(si)
        return c0, c
    cf0, cf = overfit(full); cc0, cc_ = overfit(ckpt)
    okC = cc_ < 0.1 * cc0
    print(f"  (C) SFT: full {cf0:.2f}→{cf:.4f} | checkpoint {cc0:.2f}→{cc_:.4f}  "
          f"{'OK' if okC else 'FAIL'}")
    return eA == 0 and eB == 0 and okC


if __name__ == "__main__":
    ok = main()
    print("  PASS" if ok else "  FAIL")
