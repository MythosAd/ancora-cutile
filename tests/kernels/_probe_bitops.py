"""Probe: does cuda-tile support the bit ops needed for an on-GPU MXFP8 activation quant
kernel? Extract a float's biased exponent via bitcast→uint32, >>23, &0xFF. Also test
astype(f32→int32) and astype(f32→float8_e4m3fn). Keep (capability probe)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

def P(*a): print(*a, flush=True)

# (1) bit ops: extract biased exponent ((bits>>23)&0xFF)
try:
    @ct.kernel
    def k_exp(x, out):
        t = ct.load(x, index=(0, 0), shape=(1, 8))
        b = ct.bitcast(t, ct.uint32)
        e = (b >> 23) & 0xFF
        ct.store(out, index=(0, 0), tile=e)
    xv = np.array([[1.0, 2.0, 100.0, 0.5, 448.0, 4.0, 0.25, 1024.0]], np.float32)
    gx = GA(xv); go = GA(np.zeros((1, 8), np.uint32))
    ct.launch(si, (1, 1, 1), k_exp, (gx, go)); cudart.cudaStreamSynchronize(si)
    got = go.np()[0]; exp = (xv.view(np.uint32)[0] >> 23) & 0xFF
    P(f"(1) bit-ops exponent: got {got}  want {exp}  {'OK' if np.array_equal(got, exp) else 'FAIL'}")
except Exception as ex:
    P(f"(1) bit-ops FAILED: {type(ex).__name__}: {str(ex)[:120]}")

# (2) astype f32 -> int32 (truncation)
try:
    @ct.kernel
    def k_int(x, out):
        t = ct.load(x, index=(0, 0), shape=(1, 8))
        ct.store(out, index=(0, 0), tile=ct.astype(t, ct.int32))
    xv = np.array([[1.7, 2.9, -1.5, 6.99, 0.4, 8.0, 127.5, 3.2]], np.float32)
    gx = GA(xv); go = GA(np.zeros((1, 8), np.int32))
    ct.launch(si, (1, 1, 1), k_int, (gx, go)); cudart.cudaStreamSynchronize(si)
    P(f"(2) astype f32->int32: got {go.np()[0]}  (truncation expected)")
except Exception as ex:
    P(f"(2) astype int32 FAILED: {type(ex).__name__}: {str(ex)[:120]}")

# (3) astype f32 -> float8_e4m3fn -> uint8 (the quant step)
try:
    @ct.kernel
    def k_fp8(x, out):
        t = ct.load(x, index=(0, 0), shape=(1, 8))
        ct.store(out, index=(0, 0), tile=ct.bitcast(ct.astype(t, ct.float8_e4m3fn), ct.uint8))
    xv = np.array([[1.0, 2.0, 0.5, 448.0, -2.0, 4.0, 1.5, 3.0]], np.float32)
    gx = GA(xv); go = GA(np.zeros((1, 8), np.uint8))
    ct.launch(si, (1, 1, 1), k_fp8, (gx, go)); cudart.cudaStreamSynchronize(si)
    P(f"(3) astype f32->e4m3 bytes: got {[hex(v) for v in go.np()[0]]}  (1.0=0x38, 2.0=0x40)")
except Exception as ex:
    P(f"(3) astype e4m3 FAILED: {type(ex).__name__}: {str(ex)[:120]}")
