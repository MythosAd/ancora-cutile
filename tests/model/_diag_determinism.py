"""Which model is nondeterministic? Run each forward TWICE, check bitwise identity.
Hypothesis: host Qwen3Model churns GPU alloc/free every kernel → races at depth; device
ResidentModel uses persistent buffers → deterministic (the RL-correctness requirement)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.qwen3_model import Qwen3Model
from ancora.model.resident_model import ResidentModel
from ancora.model.load_qwen3 import load_qwen3

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = Qwen3Config(); V = 151936; B, S = 1, 128
W = load_qwen3(n_layers=NL)
ids = np.random.default_rng(0).integers(0, V, (B, S)).astype(np.int64)


def to_flat(w, nl):
    f = {"embed": w["embed"], "lm_head": w["lm_head"], "final_norm": w["final_norm"]}
    for i in range(nl):
        for n, v in w["layers"][i].items(): f[f"layer{i}.{n}"] = v
    return f


print(f"NL={NL}  (forward twice each, bitwise-identical?)")

host = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0); host.load(to_flat(W, NL))
h1, _ = host.forward(ids, si)
h2, _ = host.forward(ids, si)
print(f"  host Qwen3Model:   identical={np.array_equal(h1, h2)}   "
      f"max|h1-h2|={np.abs(h1-h2).max():.3f}  rms(h1)={np.sqrt((h1**2).mean()):.3f} rms(h2)={np.sqrt((h2**2).mean()):.3f}")

rm = ResidentModel(cfg, W, B, S, V)
d1 = rm.forward(ids, si).copy()
d2 = rm.forward(ids, si).copy()
print(f"  device ResidentModel: identical={np.array_equal(d1, d2)}   "
      f"max|d1-d2|={np.abs(d1-d2).max():.3f}  rms(d1)={np.sqrt((d1**2).mean()):.3f} rms(d2)={np.sqrt((d2**2).mean()):.3f}")
