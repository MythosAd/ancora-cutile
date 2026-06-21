"""ResidentModel — device-resident multi-layer Qwen3-0.6B training. Two device-side claims (no
host-model reference: the host-orchestrated Qwen3Model allocs/frees GPU scratch every kernel and
that churn RACES at depth → it is NONDETERMINISTIC at NL≥8, see _diag_determinism.py. The device
model uses persistent buffers, no churn → deterministic. Cross-validation that the device LAYER
math equals the host layer math lives in _diag_resident_drift.py: with the FP32 residual stream
(2026-06-03) per-layer drift vs host is 0.04-0.48% — the ~6912 massive activation no longer
accumulates bf16 rounding across layers. NB: that _diag must run the host and device chains
SEPARATELY; interleaving them on one stream races the host's alloc-churn and reports bogus drift):
  (1) DETERMINISM — forward the same input TWICE → bitwise identical. This is the RL-correctness
      requirement (rollout==training logprobs) AND the proof the device chain is sound.
  (2) TRAINS END-TO-END — overfit a fixed batch at the FULL requested depth → CE collapses; per-step
      time amortizes the fixed boundary cost (≈0.36 s/layer-step at NL=28 vs host ≈7.7 s/layer-step).

Run:  python tests/model/test_resident_model.py [n_layers] [n_steps]
"""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.load_qwen3 import load_qwen3

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def determinism_check(weights, cfg, V, NL):
    """Device ResidentModel forward twice on the same input → must be bitwise identical."""
    B, S = 1, 128
    ids = np.random.default_rng(0).integers(0, V, (B, S)).astype(np.int64)
    rm = ResidentModel(cfg, weights, B, S, V)
    d1 = rm.forward(ids, si).copy()
    d2 = rm.forward(ids, si).copy()
    same = np.array_equal(d1, d2)
    print(f"  (1) determinism (fwd ×2 @ {NL} layers):  "
          f"{'bitwise IDENTICAL' if same else 'DIFFER'}  max|Δ|={np.abs(d1-d2).max():.3g}  "
          f"{'OK' if same else 'FAIL'}")
    rm.free(); gc.collect()
    return same


def overfit_check(weights, cfg, V, NL, STEPS):
    """Full-depth SFT overfit a fixed batch → CE must collapse."""
    B, S = 1, 128
    rng = np.random.default_rng(1)
    ids = rng.integers(0, V, (B, S)).astype(np.int64)
    labels = rng.integers(0, V, B * S).astype(np.int64)
    rm = ResidentModel(cfg, weights, B, S, V)
    ces, dt = [], []
    for step in range(STEPS):
        t = time.time()
        h = rm.forward(ids, si)
        ce = rm.loss_backward(h, labels, si)
        rm.step(si, lr=2e-3)
        ces.append(ce); dt.append(time.time() - t)
        print(f"  step {step:2d}  CE = {ce:7.4f}   ({dt[-1]:.2f}s)")
    best = min(ces)
    print(f"  (2) CE {ces[0]:.3f} → {ces[-1]:.3f}  (best {best:.3f})   "
          f"{np.mean(dt):.2f}s/step = {np.mean(dt)/NL:.2f}s/layer-step  (host ≈ 7.7s/layer-step)")
    # learning = a clear multi-nat drop from the start (robust to the aggressive-lr bounce + step count;
    # CE→0 needs more steps but a >3-nat drop already proves fwd→bwd→update all work end-to-end)
    ok = best < ces[0] - 3.0
    rm.free()
    print(f"  {'OK — learns (CE collapsing)' if ok else 'FAIL'}")
    return ok


def main(NL=4, STEPS=12):
    cfg = Qwen3Config(); V = 151936
    assert cfg.head_dim == 128
    print(f"loading REAL Qwen3-0.6B ({NL} of 28 layers, V={V}, head_dim={cfg.head_dim})...")
    t0 = time.time(); weights = load_qwen3(n_layers=NL); print(f"  loaded in {time.time()-t0:.0f}s")
    print("-" * 68)
    ok = determinism_check(weights, cfg, V, NL)
    print("-" * 68)
    ok &= overfit_check(weights, cfg, V, NL, STEPS)
    print("=" * 68)
    print(f"  {'PASS — device-resident ResidentModel trains real Qwen3-0.6B' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print("ResidentModel — device-resident multi-layer real Qwen3-0.6B SFT")
    print("=" * 68)
    main(NL, STEPS)
