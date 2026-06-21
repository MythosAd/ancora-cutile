"""Isolation harness for RMSNorm kernels — find which kernel/shape hangs the
cuda-tile compiler. Run each stage with flush so partial progress is visible even
if a later stage hangs. Keep this (diagnostic for future toolchain changes).

History: full-row (TM,1024) tiles compiled pathologically slowly (fwd ok, dx hung
>150 s); dw with M_BLOCKS=256 unrolled and hung. Fix: chunk H (TH=256) + Megatron
two-pass dW. Stages below re-check the previously-failing shapes."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

def P(*a): print(*a, flush=True)
class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

stage = sys.argv[1] if len(sys.argv) > 1 else "all"
ONE = 0x3f80  # bf16 bits for 1.0

def _bfbits(x):
    u = x.astype(np.float32).view(np.uint32); u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)
def _tof32(u): return (u.astype(np.uint32) << 16).view(np.float32)

def fwd(H):
    from ancora.kernels.norm import _rmsnorm_stats, _rmsnorm_apply, TM, TH
    M = TM; t = time.time(); rng = np.random.default_rng(0)
    P(f"compile+run rmsnorm fwd (stats+apply)  M={M} H={H} (HB={H//TH}) RANDOM ...")
    xf = (rng.standard_normal((M, H)) * 0.8).astype(np.float32)
    wf = (1.0 + rng.standard_normal((1, H)) * 0.2).astype(np.float32)
    gx = GA(_bfbits(xf)); gw = GA(_bfbits(wf))
    gy = GA(np.zeros((M, H), np.uint16)); gr = GA(np.zeros((M, 1), np.float32))
    ct.launch(si, (M // TM, 1, 1), _rmsnorm_stats, (gx, gr, H // TH, 1.0 / H, 1e-6))
    ct.launch(si, (M // TM, 1, 1), _rmsnorm_apply, (gx, gw, gr, gy, H // TH))
    cudart.cudaStreamSynchronize(si)
    y = _tof32(gy.np())
    xb = _tof32(_bfbits(xf)).astype(np.float64); wb = _tof32(_bfbits(wf)).astype(np.float64)
    rstd = 1.0 / np.sqrt((xb*xb).mean(-1, keepdims=True) + 1e-6)
    yr = xb * rstd * wb
    P(f"  rstd kernel[0]={gr.np()[0,0]:.5f} ref[0]={rstd[0,0]:.5f}")
    for h in range(H // TH):
        sl = slice(h*TH, (h+1)*TH)
        e = np.abs(y[:, sl] - yr[:, sl]).max() / (np.abs(yr).max() + 1e-9)
        P(f"  chunk {h} cols[{h*TH}:{(h+1)*TH}] relerr={e*100:6.2f}%  "
          f"y[0,{h*TH}]={y[0,h*TH]:+.4f} ref={yr[0,h*TH]:+.4f}")
    P(f"  ({time.time()-t:.1f}s)  DONE")

def dx(H):
    from ancora.kernels.norm import _rmsnorm_bwd_dx, TM, TH
    M = TM; t = time.time()
    P(f"compile+run _rmsnorm_bwd_dx  M={M} H={H} (HB={H//TH}) ...")
    gx = GA(np.full((M, H), ONE, np.uint16)); gw = GA(np.full((1, H), ONE, np.uint16))
    gdy = GA(np.full((M, H), ONE, np.uint16)); gr = GA(np.ones((M, 1), np.float32))
    gdx = GA(np.zeros((M, H), np.uint16))
    ct.launch(si, (M // TM, 1, 1), _rmsnorm_bwd_dx, (gx, gw, gdy, gr, gdx, H // TH, 1.0 / H))
    cudart.cudaStreamSynchronize(si)
    P(f"  ({time.time()-t:.1f}s)  DONE dx H={H}")

def dw(M, H):
    from ancora.kernels.norm import _rmsnorm_dw_part, _rmsnorm_dw_reduce, TM, TD, PART
    MB = M // TM; BPP = (MB + PART - 1) // PART; t = time.time()
    P(f"compile+run two-pass dW  M={M} H={H}  M_BLOCKS={MB} BPP={BPP} PART={PART} ...")
    gx = GA(np.full((M, H), ONE, np.uint16)); gdy = GA(np.full((M, H), ONE, np.uint16))
    gr = GA(np.ones((M, 1), np.float32)); gpart = GA(np.zeros((PART, H), np.float32))
    gdw = GA(np.zeros((1, H), np.float32))
    ct.launch(si, (H // TD, PART, 1), _rmsnorm_dw_part, (gx, gdy, gr, gpart, MB, BPP))
    ct.launch(si, (H // TD, 1, 1), _rmsnorm_dw_reduce, (gpart, gdw))
    cudart.cudaStreamSynchronize(si)
    P(f"  dw[0,:4]={gdw.np()[0,:4]} (expect M={M})  ({time.time()-t:.1f}s)  DONE")

if stage in ("fwd", "all"): fwd(1024)
if stage in ("dx", "all"):  dx(1024)
if stage in ("dw", "all"):  dw(4096, 1024)
P("stage", stage, "complete")
