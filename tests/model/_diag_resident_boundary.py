"""Localize the ResidentModel↔Qwen3Model post-final-norm divergence (18.88% in test, but the layer
chain matches to 0.10%). Compare embed input, pre-final-norm hidden, and post-final-norm hidden."""
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
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cfg = Qwen3Config(); V = 151936; B, S, H, M = 1, 128, cfg.hidden, 128
W = load_qwen3(n_layers=NL)
rng = np.random.default_rng(0)
ids = rng.integers(0, V, (B, S)).astype(np.int64)


def to_flat(w, nl):
    flat = {"embed": w["embed"], "lm_head": w["lm_head"], "final_norm": w["final_norm"]}
    for i in range(nl):
        for n, val in w["layers"][i].items(): flat[f"layer{i}.{n}"] = val
    return flat


# MATCH test_resident_model ORDER: host built + run, THEN rm built + run
host = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0); host.load(to_flat(W, NL))
hh, hc = host.forward(ids, si)            # hh = post-final-norm (M,H)
host_pre = hc["x_pre"]                    # (M,H) pre-final-norm

rm = ResidentModel(cfg, W, B, S, V)
hd = rm.forward(ids, si)                  # post-final-norm (M,H)
rm_pre = rm._cache["hpre"]               # (M,H) pre-final-norm

_bits2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
rm_embed = _bits2f(rm.gembed.to_numpy())     # tied device embed (V,H), bf16-valued
print(f"NL={NL}")
print(f"  embed weight rel:        {rel(rm_embed, host.embed)*100:.4f}%  (tied device embed vs host)")
print(f"  final_norm identical:    {np.array_equal(host.final_norm, rm.final_norm)}")
print(f"  pre-final-norm  rel:     {rel(rm_pre, host_pre)*100:.2f}%")
print(f"  post-final-norm rel:     {rel(hd, hh)*100:.2f}%")
print(f"  host_pre rms={np.sqrt((host_pre**2).mean()):.4f}  rm_pre rms={np.sqrt((rm_pre**2).mean()):.4f}")
print(f"  hh   max={np.abs(hh).max():.3f}  hd max={np.abs(hd).max():.3f}")
print(f"  hh[:2,:4]=\n{hh[:2,:4]}\n  hd[:2,:4]=\n{hd[:2,:4]}")
