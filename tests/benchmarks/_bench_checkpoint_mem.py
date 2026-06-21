"""Measure the activation-checkpointing memory win: build ResidentMoEModel (real size) at increasing
single-sequence length M, with long_context off vs on, run one fwd+bwd+step, report VRAM. Finds the
max length each reaches on 16 GB. Fresh process per config (WDDM rule).
Usage: _bench_checkpoint_mem.py [M lc]   (no arg → sweep)"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 3:
    print(f"{'M':>6} {'long_ctx':>9} {'VRAM':>8} {'status':>10}")
    for m, lc in [(2048, 0), (4096, 0), (2048, 1), (4096, 1), (8192, 1), (16384, 1)]:
        subprocess.run([sys.executable, __file__, str(m), str(lc)])
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

M, LC = int(sys.argv[1]), bool(int(sys.argv[2]))
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
try:
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, M)
    train = ResidentMoEModel(cfg, w, 1, M, device_route=True, long_context=LC)
    ids = np.random.default_rng(0).integers(0, cfg.vocab, size=(1, M)).astype(np.int64)
    labels = np.random.default_rng(1).integers(0, cfg.vocab, size=(M,)).astype(np.int64)
    train.forward(ids, si); train.loss_backward(None, labels, si); train.step(si, 1e-4)
    cudart.cudaStreamSynchronize(si)
    free, total = cudart.cudaMemGetInfo()[1:]
    used = (total - free) / 2**30
    st = "PAGING" if used > 16.0 else "OK"
    print(f"{M:>6} {('ON' if LC else 'off'):>9} {used:>6.1f}GB {st:>10}")
except Exception as e:
    print(f"{M:>6} {('ON' if LC else 'off'):>9} {'--':>8} {'OOM/'+type(e).__name__:>10}")
