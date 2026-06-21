"""Pinpoint the fp32-residual chain bug: layer 0 gout matches host (0.57%) but feeding it to
layer 1 gives 70%. Test (a) buffer hazard: layer1(gout0-direct) vs layer1(fresh copy of gout0);
(b) where layer1 diverges from host."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env
from ancora.model.qwen3_layer import Qwen3Config, TransformerLayer, _bf
from ancora.model.resident import _DBuf
from ancora.model.resident_train import ResidentLayerTrain
from ancora.model.load_qwen3 import load_qwen3

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

cfg = Qwen3Config(); H = cfg.hidden; B, S, M = 1, 128, 128; V = 151936
W = load_qwen3(n_layers=2)
rng = np.random.default_rng(0)
ids = rng.integers(0, V, (B, S)).astype(np.int64)
x0 = _bf(W["embed"][ids.reshape(-1)].reshape(B, S, H).astype(np.float32))

hosts = [TransformerLayer(cfg, seed=0) for _ in range(2)]
for i, l in enumerate(hosts):
    for n in l.w: l.w[n] = W["layers"][i][n]
devs = [ResidentLayerTrain(cfg, W["layers"][i], B, S) for i in range(2)]

bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)

# host chain (layer 1 with cache → intermediates)
xh0 = hosts[0].forward(x0.copy(), si)
xh1, c1 = hosts[1].forward(xh0, si, return_cache=True)

# device layer 0
gx = _DBuf(np.ascontiguousarray(x0.reshape(M, H), np.float32))
gout0 = devs[0].forward(gx, si); cudart.cudaStreamSynchronize(si)
d0 = gout0.to_numpy()
print(f"layer0 device vs host: {rel(d0, xh0.reshape(M,H))*100:.2f}%   d0 max={np.abs(d0).max():.2f}")

# device layer 1 — feed gout0
g1 = devs[1].forward(gout0, si); cudart.cudaStreamSynchronize(si)
L = devs[1]
print("-- layer1 internals (device vs host cache) --")
print(f"  input gx (=d0)         : matches host x?  {rel(d0, c1['x'])*100:.2f}%")
print(f"  gh  (input_ln out, bf16): {rel(bf32(L.gh.to_numpy()), c1['h1'])*100:.2f}%   "
      f"host|h1|max={np.abs(c1['h1']).max():.2f}")
print(f"  gx2 (x + attn, fp32)   : {rel(L.gx2.to_numpy(), c1['x2'])*100:.2f}%   "
      f"host|x2|max={np.abs(c1['x2']).max():.2f} dev|x2|max={np.abs(L.gx2.to_numpy()).max():.2f}")
print(f"  gh2 (post_ln out, bf16): {rel(bf32(L.gh2.to_numpy()), c1['h2'])*100:.2f}%")
print(f"  gout (final, fp32)     : {rel(g1.to_numpy(), xh1.reshape(M,H))*100:.2f}%")
