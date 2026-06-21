"""Probe: cuda-tile capabilities needed by DEVICE-POSITION decode kernels (per-token CUDA graph —
host at_pos pointers / runtime args freeze in a captured graph, so `pos` must live in device memory):
  A: STORE at a loaded-scalar row index            (_append_kv sidestepped this — "open question")
  B: bit ops on a loaded scalar (pos>>6, pos&63)   (block/row split + pow2 ring modulo)
  C: loop bound from a loaded scalar               (range(q_blk+1) with q_blk = load(gpos)>>6)
All three verified → the decode step can be captured once and replayed for every position."""
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
def sync(): cudart.cudaStreamSynchronize(si)


@ct.kernel
def _pa(P, X, Y):
    """Y[pos] = X[0] — store row index from a loaded scalar."""
    pos = ct.reshape(ct.load(P, index=(0, 0), shape=(1, 1)), ())
    ct.store(Y, index=(pos, 0), tile=ct.load(X, index=(0, 0), shape=(1, 64)))


@ct.kernel
def _pb(P, Y):
    """Y[pos>>2] = 1.0 row;  Y[8 + (pos&3)] = 2.0 row — scalar shift/and as indices."""
    pos = ct.reshape(ct.load(P, index=(0, 0), shape=(1, 1)), ())
    ct.store(Y, index=(pos >> 2, 0), tile=ct.full((1, 64), 1.0, ct.float32))
    ct.store(Y, index=(8 + (pos & 3), 0), tile=ct.full((1, 64), 2.0, ct.float32))


@ct.kernel
def _pc(P, X, Y):
    """Y[0] = Σ_{kv ≤ pos>>2} X[kv] — loop bound from a loaded scalar."""
    pos = ct.reshape(ct.load(P, index=(0, 0), shape=(1, 1)), ())
    q_blk = pos >> 2
    acc = ct.zeros((1, 64), ct.float32)
    for kv in range(q_blk + 1):
        acc = acc + ct.load(X, index=(kv, 0), shape=(1, 64))
    ct.store(Y, index=(0, 0), tile=acc)


def run():
    gp = _GpuArray(np.array([[13]], np.int32))            # pos = 13 → pos>>2 = 3, pos&3 = 1
    X = np.arange(8, dtype=np.float32)[:, None] * np.ones((1, 64), np.float32) + 1.0
    gX = _GpuArray(X)

    gY = _GpuArray.zeros((16, 64), np.float32)
    try:
        ct.launch(si, (1, 1, 1), _pa, (gp, gX, gY)); sync()
        y = gY.to_numpy()
        ok = y[13, 0] == 1.0 and np.abs(y).sum() == 64.0
        print(f"  A scalar STORE index : {'OK' if ok else 'WRONG ' + str(np.nonzero(y[:,0])[0])}")
    except Exception as e:
        print(f"  A scalar STORE index : FAILED — {type(e).__name__}: {str(e)[:150]}")

    gY2 = _GpuArray.zeros((16, 64), np.float32)
    try:
        ct.launch(si, (1, 1, 1), _pb, (gp, gY2)); sync()
        y = gY2.to_numpy()
        ok = y[3, 0] == 1.0 and y[9, 0] == 2.0 and np.abs(y[:, 0]).sum() == 3.0
        print(f"  B scalar >> & ops    : {'OK' if ok else 'WRONG ' + str(np.nonzero(y[:,0])[0])}")
    except Exception as e:
        print(f"  B scalar >> & ops    : FAILED — {type(e).__name__}: {str(e)[:150]}")

    gY3 = _GpuArray.zeros((1, 64), np.float32)
    try:
        ct.launch(si, (1, 1, 1), _pc, (gp, gX, gY3)); sync()
        got = float(gY3.to_numpy()[0, 0])
        ok = got == 1 + 2 + 3 + 4                          # rows 0..3
        print(f"  C scalar loop bound  : {'OK' if ok else f'WRONG got={got} want=10'}")
    except Exception as e:
        print(f"  C scalar loop bound  : FAILED — {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    print("Probe: device-position decode primitives (scalar store-index / bit-ops / loop bound)")
    run()
