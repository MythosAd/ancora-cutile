"""Profile the Muon optimizer step components at real size (M=2048): batched square proj NS vs
the per-layer step (q/o per-weight muon + norms AdamW + MoE expert NS) vs embed/final-norm AdamW.
Finds where the ~226ms goes so we know what to batch next."""
import sys, os, time
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
def sync(): cudart.cudaStreamSynchronize(si)

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, 1, M)
m = ResidentMoEModel(cfg, w, 1, M, device_route=True, optimizer="muon")
ids = np.random.default_rng(0).integers(0, cfg.vocab, (1, M)).astype(np.int64)
labels = np.random.default_rng(1).integers(0, cfg.vocab, (M,)).astype(np.int64)
m.forward(ids, si); m.loss_backward(None, labels, si); sync()          # populate grads (and warm)


def timed(fn, n=5):
    fn(); sync()                                                       # warm
    t = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - t) / n * 1e3


ml = m.muon_lr
t_bproj = timed(lambda: m.batched_proj_muon.step(si, ml))
t_layers = timed(lambda: [l.step(si, 1e-4, 0.9, 0.999, 1e-8, 0.0, ml) for l in m.layers])
# isolate the expert NS (MoE layers' moe.step) vs the norm part of the layer step
moe_layers = [l for l in m.layers if hasattr(l, "moe")]
t_experts = timed(lambda: [l.moe.step(si, 1e-4, muon_lr=ml) for l in moe_layers])
nw = {k: len(ws) for k, ws in m.batched_proj_muon.groups.items()}
print(f"  M={M} Muon optimizer breakdown (real size NL={cfg.n_layers}):")
print(f"    batched proj NS (ALL 2D: square + rect q/o): {t_bproj:6.1f} ms  groups {nw}")
print(f"    per-layer step (norms AdamW + expert NS): {t_layers:6.1f} ms")
print(f"      └ of which MoE expert NS ({len(moe_layers)} layers × 3): {t_experts:6.1f} ms")
print(f"    ⇒ norms AdamW ≈ {t_layers - t_experts:6.1f} ms")
