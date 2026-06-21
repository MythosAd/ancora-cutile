"""Real-size decode throughput vs Bp (fresh process per Bp — WDDM paging rule).
Usage: python _bench_decode_bp.py [Bp]   (no arg → run 32 and 64 in subprocesses)"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 2:
    for bp in (32, 64):
        subprocess.run([sys.executable, __file__, str(bp)])
    sys.exit(0)
MX = "mx" in sys.argv[2:]                            # MXFP8 forward (trainer + engine)

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

Bp = int(sys.argv[1])
cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
P, NEW, maxS = 512, 64, 1024
host = MoEModel(cfg, seed=5, grouped=False, tie=True)
w = from_host(host, 1, 128)
train = ResidentMoEModel(cfg, w, 1, 128, device_route=True, mxfp8=MX)
eng = ResidentMoEDecodeModel(train, Bp=Bp, maxS=maxS, si=si)
rng = np.random.default_rng(0)
prompts = rng.integers(0, cfg.vocab, size=(Bp, P)).astype(np.int64)
eng.generate(prompts, 4, si)
eng.generate(prompts, 4, si, so=so, dev=dev, use_graph=True)
ntok = P + NEW - 1
t0 = time.perf_counter(); eng.generate(prompts, NEW, si); dt = (time.perf_counter() - t0) / ntok
t0 = time.perf_counter(); eng.generate(prompts, NEW, si, so=so, dev=dev, use_graph=True)
dg = (time.perf_counter() - t0) / ntok
free, total = cudart.cudaMemGetInfo()[1:]
print(f"  Bp={Bp:3d}{' MX' if MX else ''}: direct {dt*1e3:.2f} → graph {dg*1e3:.2f} ms/step = "
      f"{Bp/dg:.0f} tok/s  (VRAM {(total-free)/2**30:.1f} GB)")
