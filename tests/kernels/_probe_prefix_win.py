"""Probe: cuda-tile runtime-scalar conditional forms needed by the WINDOWED prefix attention kernels.
The window can span the prefix/suffix boundary, so a loop iteration needs BOTH `kv >= 0` AND
`kv < NB` (two runtime conditions; existing kernels only use one). Tests which guard forms compile
and compute correctly:
  A: `if (kv >= 0) and (kv < NB):`   — python and
  B: nested `if kv >= 0:` / `if kv < NB:`
  C: `if (kv >= 0) & (kv < NB):`     — bitwise &
Expected per output row b: sum of X rows [b-WB, min(b, NB-1)] ∩ [0, NB)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.tile as ct
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.loss import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

NB, WB, NQ = 4, 6, 8   # 4 source blocks, window 6, 8 query blocks → kv ranges hit both edges


@ct.kernel
def _pa(X, Y, NB: ct.Constant[int], WB: ct.Constant[int]):
    b = ct.bid(0)
    acc = ct.zeros((1, 64), ct.float32)
    for w in range(WB + 1):
        kv = b - WB + w
        if (kv >= 0) and (kv < NB):
            acc = acc + ct.load(X, index=(kv, 0), shape=(1, 64))
    ct.store(Y, index=(b, 0), tile=acc)


@ct.kernel
def _pb(X, Y, NB: ct.Constant[int], WB: ct.Constant[int]):
    b = ct.bid(0)
    acc = ct.zeros((1, 64), ct.float32)
    for w in range(WB + 1):
        kv = b - WB + w
        if kv >= 0:
            if kv < NB:
                acc = acc + ct.load(X, index=(kv, 0), shape=(1, 64))
    ct.store(Y, index=(b, 0), tile=acc)


@ct.kernel
def _pc(X, Y, NB: ct.Constant[int], WB: ct.Constant[int]):
    b = ct.bid(0)
    acc = ct.zeros((1, 64), ct.float32)
    for w in range(WB + 1):
        kv = b - WB + w
        if (kv >= 0) & (kv < NB):
            acc = acc + ct.load(X, index=(kv, 0), shape=(1, 64))
    ct.store(Y, index=(b, 0), tile=acc)


def run(name, kern):
    X = np.arange(NB, dtype=np.float32)[:, None] * np.ones((1, 64), np.float32) + 1.0  # row i = i+1
    ref = np.zeros((NQ, 64), np.float32)
    for b in range(NQ):
        for kv in range(max(0, b - WB), min(b, NB - 1) + 1):
            ref[b] += X[kv]
    gX = _GpuArray(X); gY = _GpuArray.zeros((NQ, 64), np.float32)
    try:
        ct.launch(si, (NQ, 1, 1), kern, (gX, gY, NB, WB))
        cudart.cudaStreamSynchronize(si)
        out = gY.to_numpy()
        err = float(np.abs(out - ref).max())
        print(f"  {name}: {'OK exact' if err == 0.0 else f'WRONG err={err}'}  (row sums: {out[:,0].tolist()})")
        return err == 0.0
    except Exception as e:
        print(f"  {name}: FAILED to compile/run — {type(e).__name__}: {str(e)[:200]}")
        return False


if __name__ == "__main__":
    print("Probe: compound runtime-scalar guards in cuda-tile loops (needed by windowed prefix attn)")
    print(f"  expect row b = sum of rows [b-{WB}, min(b,{NB-1})]")
    run("A `and`  ", _pa)
    run("B nested ", _pb)
    run("C `&`    ", _pc)
