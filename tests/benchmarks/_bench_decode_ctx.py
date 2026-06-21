"""Rollout KV-cache memory vs (Bp, context length). Isolates the decode engine's KV+buffer cost
(VRAM after building the engine − VRAM of the trainer alone) and a short-generate tok/s, per
(Bp, maxS). Shows the local-RING (O(window), fixed) vs global-FULL (O(maxS·Bp)) split — so longer
context costs only via the 2 global layers. Fresh process per point (WDDM rule).
Usage: _bench_decode_ctx.py [Bp maxS]   (no args → sweep)"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POINTS = [(64, 1024), (64, 2048), (64, 4096), (32, 4096), (32, 8192), (16, 8192), (16, 16384)]
if len(sys.argv) < 3:
    print(f"{'Bp':>4} {'maxS':>6} {'KV+buf GB':>10} {'localKV':>8} {'globalKV':>9} {'totVRAM':>8} {'tok/s':>7}")
    for bp, ms in POINTS:
        subprocess.run([sys.executable, __file__, str(bp), str(ms)])
    sys.exit(0)

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

Bp, maxS = int(sys.argv[1]), int(sys.argv[2])
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, 1, 128)
train = ResidentMoEModel(cfg, w, 1, 128, device_route=True)
free0 = cudart.cudaMemGetInfo()[1]                       # VRAM free with trainer only
eng = ResidentMoEDecodeModel(train, Bp=Bp, maxS=maxS, si=si)
free1, total = cudart.cudaMemGetInfo()[1:]
kv_gb = (free0 - free1) / 2**30                          # the engine's KV + scratch

# analytic split (bf16 cache): global = full maxS, local = ring (O(window))
Hkv, Dh, BKV = cfg.n_kv_heads, cfg.head_dim, 64
n_glob = sum(1 for l in eng.layers if l.is_global); n_loc = cfg.n_layers - n_glob
crows = eng.layers[[i for i,l in enumerate(eng.layers) if not l.is_global][0]].CROWS
glob = n_glob * Bp * Hkv * maxS  * Dh * 2 * 2 / 2**30    # K+V, 2 bytes
loc  = n_loc  * Bp * Hkv * crows * Dh * 2 * 2 / 2**30

prompts = np.random.default_rng(0).integers(0, cfg.vocab, size=(Bp, 32)).astype(np.int64)
eng.generate(prompts, 8, si); eng.generate(prompts, 8, si, so=so, dev=dev, use_graph=True)  # warm
t0 = time.perf_counter(); eng.generate(prompts, 32, si, so=so, dev=dev, use_graph=True)
toks = Bp * 32 / (time.perf_counter() - t0)
print(f"{Bp:>4} {maxS:>6} {kv_gb:>10.2f} {loc:>8.2f} {glob:>9.2f} {(total-free1)/2**30:>8.1f} {toks:>7.0f}")
