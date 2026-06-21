"""Debug: resident PrefixGlobalAttn(window=W) vs the validated host helpers, failing geometry
Sp=256 Sc=128 W=128 (NKVBp > win_blocks). Isolates resident-class wiring from kernels/layer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.attention import flash_attn_forward_prefix, flash_attn_backward_prefix, D
from ancora.rl.prefix_resident import PrefixGlobalAttn

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def maxabs(a, b): return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())

G, Hq, Hkv, Sp, Sc, W = 4, 16, 8, 256, 128, 128
rng = np.random.default_rng(7)
pq = (rng.standard_normal((Hq,  Sp, D)) * 0.5).astype(np.float32)
pk = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
pv = (rng.standard_normal((Hkv, Sp, D)) * 0.5).astype(np.float32)
sq = (rng.standard_normal((G, Hq,  Sc, D)) * 0.5).astype(np.float32)
sk = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
sv = (rng.standard_normal((G, Hkv, Sc, D)) * 0.5).astype(np.float32)
dOp = (rng.standard_normal((Hq, Sp, D)) * 0.3).astype(np.float32)
dOs = (rng.standard_normal((G, Hq, Sc, D)) * 0.3).astype(np.float32)

pa = PrefixGlobalAttn(Hq, Hkv, D, Sp, Sc, G, window=W)
Op, Os = pa.forward(pq, pk, pv, sq, sk, sv, si)
grad = pa.backward(Op, Os, dOp, dOs, si)

hOp, hOs, hLp, hLs = flash_attn_forward_prefix(pq, pk, pv, sq, sk, sv, si, window=W)
print(f"fwd  Op Δ={maxabs(Op, hOp):.2e}  Os Δ={maxabs(Os, hOs):.2e}")
hg = flash_attn_backward_prefix(pq, pk, pv, sq, sk, sv, hOp, hOs, hLp, hLs, dOp, dOs, si, window=W)
names = ["dQp", "dKp", "dVp", "dQs", "dKs", "dVs"]
for j in range(6):
    print(f"bwd  {names[j]} Δ={maxabs(grad[j], hg[j]):.2e}")
