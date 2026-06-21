"""Push long_context past 16K with the Muon-freed optimizer VRAM. Build the REAL-size MoE model
(ResidentMoEModel, vocab=151936, NL=12) at increasing single-sequence length M, long_context=True,
with optimizer adamw vs muon; run one fwd+bwd+step; report VRAM + whether it fits 16 GB. The Muon
hybrid drops the experts' + proj's v buffer (the AdamW floor that caps the max length); this finds
the new max each optimizer reaches. Fresh process per config (WDDM paging rule — a co-resident model
silently pages and corrupts the timing/VRAM read).
Usage: _bench_muon_longctx.py [M optimizer]   (no arg → sweep)"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 3:
    print(f"{'M(seq)':>7} {'optim':>6} {'VRAM':>8} {'status':>9}")
    # 8K both fit (baseline); 16K is the AdamW edge; 20-24K is what Muon should unlock
    for m in (8192, 16384, 20480, 24576):
        for opt in ("adamw", "muon"):
            subprocess.run([sys.executable, __file__, str(m), opt])
    sys.exit(0)

import time
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

M, OPT = int(sys.argv[1]), sys.argv[2]
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
phase = "construct"
try:
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, M)
    train = ResidentMoEModel(cfg, w, 1, M, device_route=True, long_context=True, optimizer=OPT)
    free_c = cudart.cudaMemGetInfo()[1]                  # free AFTER construct, BEFORE the lazy opt alloc
    ids = np.random.default_rng(0).integers(0, cfg.vocab, size=(1, M)).astype(np.int64)
    labels = np.random.default_rng(1).integers(0, cfg.vocab, size=(M,)).astype(np.int64)
    def stepf():
        train.forward(ids, si); train.loss_backward(None, labels, si); train.step(si, 1e-4)
        cudart.cudaStreamSynchronize(si)
    phase = "step"
    stepf()                                              # warmup (JIT + lazy optimizer alloc — the opt VRAM)
    free, total = cudart.cudaMemGetInfo()[1:]
    used = (total - free) / 2**30; cons = (total - free_c) / 2**30
    dts = []
    for _ in range(3):
        t = time.perf_counter(); stepf(); dts.append(time.perf_counter() - t)
    ms = np.median(dts) * 1e3
    st = "PAGING" if (used > 15.95 and ms > 6000) else "OK"
    print(f"{M:>7} {OPT:>6} construct {cons:>5.2f}GB +step {used:>5.2f}GB {ms:>7.0f}ms {st:>8}")
except Exception as e:
    print(f"{M:>7} {OPT:>6} OOM@{phase} ({type(e).__name__})")
