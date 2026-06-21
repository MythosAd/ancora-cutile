"""Megakernel fusion #1: _gateup_swiglu_q (gate/up MXFP8 GEMM + SwiGLU + quant in one
kernel) vs the separate 3-kernel path (gate GEMM, up GEMM, swiglu_q). Proves correctness
(dequant ≈ host silu(gate)*up) and that fusing away the gate/up HBM round-trip (96MB write
+ 96MB read) is faster. Keep."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _gateup_swiglu_q
from ancora.kernels.linear import _fwd_mxfp8_bf16, mxfp8_tile
from ancora.kernels.activation import _swiglu_fwd_q, QB
from ancora.kernels.quant import _quant_mxfp8, QTM, B as B32, quantize_colblock, dequantize_rowblock

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
bfv = lambda x: (bf(x).astype(np.uint32) << 16).view(np.float32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)

class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

def tmms(launch, it=50, wm=12):
    for _ in range(wm): launch()
    so.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, SI)
    for _ in range(it): launch()
    cudart.cudaEventRecord(t1, SI); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / it


def setup(M, H, I):
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
    wg = (rng.standard_normal((H, I)) * 0.04).astype(np.float32)
    wu = (rng.standard_normal((H, I)) * 0.04).astype(np.float32)
    gh = GA(bf(h)); ghf = GA.z((M, H), np.uint8); ghs = GA.z((M, H // B32), np.uint8)
    ct.launch(SI, (M // QTM, 1, 1), _quant_mxfp8, (gh, ghf, ghs, H // B32))
    wgf, wgs = quantize_colblock(wg); wuf, wus = quantize_colblock(wu)
    gwgf = GA(wgf); gwgs = GA(wgs); gwuf = GA(wuf); gwus = GA(wus)
    cudart.cudaStreamSynchronize(SI)
    # host reference: gate=hq@wgq, up=hq@wuq, y=silu(gate)*up (use dequant operands)
    hq = dequantize_rowblock(ghf.np(), ghs.np())
    wgq = dequantize_rowblock(wgf.T, wgs.T).T   # colblock dequant via row helper on transpose
    wuq = dequantize_rowblock(wuf.T, wus.T).T
    gate = hq.astype(np.float64) @ wgq.astype(np.float64)
    up = hq.astype(np.float64) @ wuq.astype(np.float64)
    yref = (gate * (0.5 + 0.5 * np.tanh(0.5 * gate))) * up
    return (ghf, ghs, gwgf, gwgs, gwuf, gwus), yref


if __name__ == "__main__":
    print("Megakernel fusion #1: gate/up + SwiGLU + quant"); print("=" * 64)
    # correctness at small M
    M, H, I = 256, 1024, 3072
    (ghf, ghs, gwgf, gwgs, gwuf, gwus), yref = setup(M, H, I)
    af = GA.z((M, I), np.uint8); asc = GA.z((M, I // QB), np.uint8)
    TM, TK = 64, 128
    ct.launch(SI, (M // TM, I // 32, 1), _gateup_swiglu_q, (ghf, ghs, gwgf, gwgs, gwuf, gwus, af, asc, H // TK, TM, TK))
    cudart.cudaStreamSynchronize(SI)
    deq = dequantize_rowblock(af.np(), asc.np())
    e = rel(deq, yref); print(f"  correctness (dequant vs host swiglu): {e*100:.2f}%  {'OK' if e < 0.08 else 'FAIL'}")

    # perf at real size
    M, H, I = 8192, 1024, 3072
    (ghf, ghs, gwgf, gwgs, gwuf, gwus), _ = setup(M, H, I)
    gg = GA.z((M, I), np.uint16); gu = GA.z((M, I), np.uint16)
    af = GA.z((M, I), np.uint8); asc = GA.z((M, I // QB), np.uint8)
    af2 = GA.z((M, I), np.uint8); asc2 = GA.z((M, I // QB), np.uint8)
    TMg, TNg, TKg = mxfp8_tile(I, H)
    def separate():
        ct.launch(SI, (M // TMg, I // TNg, 1), _fwd_mxfp8_bf16, (ghf, gwgf, ghs, gwgs, gg, H // TKg, TMg, TNg, TKg))
        ct.launch(SI, (M // TMg, I // TNg, 1), _fwd_mxfp8_bf16, (ghf, gwuf, ghs, gwus, gu, H // TKg, TMg, TNg, TKg))
        ct.launch(SI, (M // 64, 1, 1), _swiglu_fwd_q, (gg, gu, af2, asc2, I // QB))
    print(f"  separate (gate+up GEMM, then swiglu_q):  {tmms(separate)*1000:6.0f} us")
    for TM, TK in [(64, 128), (128, 128), (64, 256), (128, 256), (64, 64)]:
        if H % TK: continue
        f = lambda TM=TM, TK=TK: ct.launch(SI, (M // TM, I // 32, 1), _gateup_swiglu_q, (ghf, ghs, gwgf, gwgs, gwuf, gwus, af, asc, H // TK, TM, TK))
        try:
            t = tmms(f)
            print(f"  fused TM={TM:3d} TK={TK:3d}:                       {t*1000:6.0f} us")
        except Exception as ex:
            print(f"  fused TM={TM:3d} TK={TK:3d}: FAIL {type(ex).__name__} {str(ex)[:60]}")
    print("=" * 64)
