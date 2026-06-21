"""Probe: does cuda-tile support the integer ops needed for an in-kernel stochastic-rounding
dither (arange, broadcast, uint32 multiply/xor/shift)? And verify SR-to-bf16 is unbiased."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

class G:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

TM, TN = 32, 64

@ct.kernel
def _sr_store(x, out, seed):
    """out = stochastic-round(x) to bf16 bits, dither = hash(global row,col,seed). Grid (M//TM, N//TN)."""
    mb, nb = ct.bid(0), ct.bid(1)
    mu = ct.astype(mb, ct.uint32); nu = ct.astype(nb, ct.uint32); su = ct.astype(seed, ct.uint32)
    row = ct.broadcast_to(ct.reshape(ct.arange(TM, dtype=ct.uint32), (TM, 1)), (TM, TN)) + mu * TM
    col = ct.broadcast_to(ct.reshape(ct.arange(TN, dtype=ct.uint32), (1, TN)), (TM, TN)) + nu * TN
    key = (row * 73856093) ^ (col * 19349663) ^ su
    h = key * 0x9E3779B1
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    dither = h & 0xFFFF
    u = ct.bitcast(ct.load(x, index=(mb, nb), shape=(TM, TN)), ct.uint32)
    ct.store(out, index=(mb, nb), tile=ct.astype((u + dither) >> 16, ct.uint16))


def main():
    M, N = 64, 128
    # x between two bf16 grid points so SR straddles: e.g. value whose low-16 bits are ~half
    x = np.full((M, N), 1.0 + 1.0 / 256.0 * 0.4, np.float32)   # 0.4 of the way to next bf16 step
    gx = G(x)
    bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    acc = np.zeros((M, N), np.float64)
    NS = 400
    for s in range(NS):
        go = G(np.zeros((M, N), np.uint16))
        ct.launch(si, (M // TM, N // TN, 1), _sr_store, (gx, go, int(s)))
        cudart.cudaStreamSynchronize(si)
        acc += bf32(go.np())
    mean = acc / NS
    # bf16 grid points around x: floor and ceil
    lo = bf32(np.full((M, N), x, np.float32).view(np.uint32) >> 16)   # truncated bf16
    print(f"x = {x[0,0]:.6f}")
    print(f"bf16 floor (trunc) = {lo[0,0]:.6f}")
    print(f"SR mean over {NS} seeds = {mean.mean():.6f}   (should ≈ x = {x[0,0]:.6f}, NOT the floor)")
    err = abs(mean.mean() - x[0,0]) / (1.0/256.0)
    print(f"  bias = {err*100:.1f}% of one bf16 ULP   {'OK — unbiased' if err < 0.1 else 'biased?'}")
    # also: a value exactly ON a bf16 grid point must round to itself always
    x2 = np.ones((M, N), np.float32); g2 = G(x2); o2 = G(np.zeros((M, N), np.uint16))
    ct.launch(si, (M // TM, N // TN, 1), _sr_store, (g2, o2, 7)); cudart.cudaStreamSynchronize(si)
    exact = np.array_equal(bf32(o2.np()), x2)
    print(f"  exact bf16 value (1.0) rounds to itself: {exact}  {'OK' if exact else 'FAIL'}")


if __name__ == "__main__":
    main()
