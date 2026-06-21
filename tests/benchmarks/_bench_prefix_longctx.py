"""RL TRAINING-side (prefix GRPO) peak VRAM, long_context off vs on, at a ~16K context. Confirms B:
checkpointing lets the GRPO training step reach the long context (≈ the SFT path's 16K wall) instead
of OOMing at full-store. Fresh process per config (WDDM rule). Usage: _bench_prefix_longctx.py Sp Sc G lc"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 5:
    print(f"{'Sp':>6}{'Sc':>6}{'G':>3} {'M':>6} {'long_ctx':>9} {'VRAM':>9} {'status':>8}")
    for Sp, Sc, G in [(2048, 512, 8), (8192, 1024, 8)]:        # M = 6144 ; 16384
        for lc in (0, 1):
            subprocess.run([sys.executable, __file__, str(Sp), str(Sc), str(G), str(lc)])
    sys.exit(0)

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import from_host
from ancora.model.resident_prefix_model import ResidentPrefixMoEModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

Sp, Sc, G, LC = (int(sys.argv[i]) for i in range(1, 5))
M = Sp + G * Sc
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
phase = "construct"
try:
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, Sp + Sc)
    pre = ResidentPrefixMoEModel(cfg, w, Sp, Sc, G, device_route=True, long_context=bool(LC))
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, cfg.vocab, size=(Sp,)).astype(np.int64)
    comps = rng.integers(0, cfg.vocab, size=(G, Sc)).astype(np.int64)
    adv = ((np.arange(G) == 0).astype(np.float32) - 1.0 / G)
    import time
    phase = "step"
    def stepf():
        h = pre.forward_prefix(prompt, comps, si)
        pre.grpo_loss_backward(h, comps, adv, si)
        pre.step(si, 2e-3); cudart.cudaStreamSynchronize(si)
    stepf()                                              # warm
    free, total = cudart.cudaMemGetInfo()[1:]; used = (total - free) / 2**30
    dts = []
    for _ in range(3):
        t = time.perf_counter(); stepf(); dts.append(time.perf_counter() - t)
    ms = np.median(dts) * 1e3
    st = "PAGING" if (used > 15.95 and ms > 8000) else "OK"   # WDDM paging ⇒ ~10× slower
    print(f"{Sp:>6}{Sc:>6}{G:>3} {M:>6} {('ON' if LC else 'off'):>9} {used:>7.2f}GB {ms:>7.0f}ms {st:>8}")
except Exception as e:
    print(f"{Sp:>6}{Sc:>6}{G:>3} {M:>6} {('ON' if LC else 'off'):>9} {'--':>9} {'--':>7} OOM@{phase}")
