"""RL-side peak VRAM at a 16K sequence: the ROLLOUT (decode engine) footprint. The decode KV-cache
is allocated at maxS up front, so the 16K peak is reached at construction — no need to actually
generate 16K tokens. Reports the trainer-weights baseline (shared, zero-copy with training) + the
rollout's marginal cost (global-layer KV-cache O(maxS) + local ring O(window) + decode scratch).
Usage: _bench_rl_vram.py [maxS Bp]"""
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
from ancora.model.resident_moe_decode import ResidentMoEDecodeModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def used(): cudart.cudaStreamSynchronize(si); f, t = cudart.cudaMemGetInfo()[1:]; return (t - f) / 2**30

maxS = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
Bp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)   # the real schedule (5L:1G, D/M)
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, 1, 128)
u0 = used()
train = ResidentMoEModel(cfg, w, 1, 128, device_route=True)         # minimal trainer = the weight host
u_w = used()
eng = ResidentMoEDecodeModel(train, Bp=Bp, maxS=maxS, si=si)        # allocs the maxS KV-cache
P = min(256, maxS - 8)
prompts = np.random.default_rng(0).integers(0, cfg.vocab, size=(Bp, P)).astype(np.int64)
eng.generate(prompts, 4, si)                                        # a few steps (cache already at maxS)
u_r = used()
print(f"  ROLLOUT @ maxS={maxS} Bp={Bp}:  weights {u_w-u0:.2f}GB  + rollout KV/scratch {u_r-u_w:.2f}GB"
      f"  = total {u_r:.2f}GB peak")
