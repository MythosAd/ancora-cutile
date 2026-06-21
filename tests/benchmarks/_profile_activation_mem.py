"""Profile where the training VRAM goes — isolate the per-layer activation cost (the checkpointing
target) from the fixed cost (AdamW + weights + boundary). Build ResidentMoEModel at varying (NL, M)
and read cudaMemGetInfo deltas. Fresh process. Usage: _profile_activation_mem.py [NL M]"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 3:
    print("  (NL, M)  modelVRAM  | derive: per-layer activation (NL-scaling), boundary+AdamW+weights (fixed)")
    for nl, m in [(4, 1024), (8, 1024), (4, 2048), (8, 2048)]:
        subprocess.run([sys.executable, __file__, str(nl), str(m)])
    sys.exit(0)

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

NL, M = int(sys.argv[1]), int(sys.argv[2])
cfg = MoEConfig(vocab=151936, n_layers=NL, period=2, window=512)   # period=2 → half global (stress)
B, S = 1, M
base_free = cudart.cudaMemGetInfo()[1]
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, B, S)
train = ResidentMoEModel(cfg, w, B, S, device_route=True)
# one fwd+bwd to allocate any lazy buffers (MoE pack, grads)
ids = np.random.default_rng(0).integers(0, cfg.vocab, size=(B, S)).astype(np.int64)
labels = np.random.default_rng(1).integers(0, cfg.vocab, size=(M,)).astype(np.int64)
train.forward(ids, si); train.loss_backward(None, labels, si); train.step(si, 1e-4)
cudart.cudaStreamSynchronize(si)
free, total = cudart.cudaMemGetInfo()[1:]
used = (base_free - free) / 2**30
nglob = sum(1 for is_g, fd in host.sched if is_g)
print(f"  NL={NL:2d} M={M:5d} (glob={nglob}): model VRAM {used:5.2f} GB")
