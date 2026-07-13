"""Full training step (fwd + loss_backward + optimizer step) muon-vs-adamw context for task #31.
One mode per process: python _bench_muon_fullstep.py adamw|muon [M]"""
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

mode = sys.argv[1] if len(sys.argv) > 1 else "adamw"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, 1, M)
m = ResidentMoEModel(cfg, w, 1, M, device_route=True, optimizer=mode)
ids = np.random.default_rng(0).integers(0, cfg.vocab, (1, M)).astype(np.int64)
labels = np.random.default_rng(1).integers(0, cfg.vocab, (M,)).astype(np.int64)

def step():
    h = m.forward(ids, si)
    ce = m.loss_backward(h, labels, si)
    m.step(si, lr=1e-4); sync()
    return ce

step(); step()                                              # warm (JIT + lazy opt state)
best = 1e9
for _ in range(3):
    t = time.perf_counter()
    for _ in range(5): step()
    best = min(best, (time.perf_counter() - t) / 5 * 1e3)
print(f"[{mode}] M={M}: full step {best:.1f} ms  ({M / best * 1e3:.0f} tok/s)")
