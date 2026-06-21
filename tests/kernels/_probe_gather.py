"""Probe: can a cuda-tile kernel read an int from an array and use it as a runtime
row-block index (data-dependent gather)? This decides the grouped-MoE kernel design:
  works  → efficient variable-group grouped GEMM (per-tile expert id from an array)
  fails  → fixed-capacity grouped GEMM (expert = block // tiles_per_expert, pure index math)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env
from ancora.kernels.attention import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


@ct.kernel
def _gather_via_array(idx, W, out):
    """out[mi] = W[ idx[mi] ]  — read an expert id from idx, index W with it."""
    mi = ct.bid(0)
    e = ct.load(idx, index=(mi, 0), shape=(1, 1))      # (1,1) int32 tile
    e_s = ct.reshape(e, ())                             # try: tile -> scalar
    wt = ct.load(W, index=(e_s, 0), shape=(1, 8))
    ct.store(out, index=(mi, 0), tile=wt)


def main():
    idx = _GpuArray(np.array([[2], [0], [3]], np.int32))
    W   = _GpuArray(np.arange(4 * 8).reshape(4, 8).astype(np.float32))
    out = _GpuArray(np.zeros((3, 8), np.float32))
    try:
        ct.launch(si, (3, 1, 1), _gather_via_array, (idx, W, out))
        cudart.cudaStreamSynchronize(si)
        r = out.to_numpy()
        exp = np.stack([np.arange(4 * 8).reshape(4, 8)[i] for i in (2, 0, 3)]).astype(np.float32)
        print("got:\n", r)
        print("PASS" if np.array_equal(r, exp) else "WRONG VALUES")
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:400])


if __name__ == "__main__":
    main()
