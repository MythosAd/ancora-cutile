"""Probe FP8 backward (Dgrad) on sm_120a: does ct.mma_scaled accept E5M2 elements, and how
does E4M3 vs E5M2 quantization error compare under E8M0 per-32-block scaling?

Dgrad = dy @ Wᵀ. Here A=dy (gradient, wide dynamic range), B=Wᵀ. A is quantized per-32-block
along K (axis 1 → scale (M,K//32)); B per-32-block along K (axis 0 → scale (K//32,N)); then
ct.mma_scaled, compared to the original f32 operands matmul'd in fp64.

Thesis: with FINE per-32-block scaling the dynamic range is handled at the SCALE layer, so the
more-precise E4M3 (3 mantissa) should BEAT E5M2 (2 mantissa); E5M2's wide element range only
pays off when a block has large within-block spread."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.loss import _GpuArray

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

B = 32; QTM = 64; QWN = 128; OFF = 7   # E8M0 offset: block-max → 2^7..2^8 (under E4M3 448 & E5M2 57344)


def _quant_row(elem):     # A(M,K) f32 → fp8(M,K) + e8m0(M,K//32); block of 32 along K (axis 1)
    @ct.kernel
    def _q(x, fp8_out, scale_out, KB: ct.Constant[int]):
        m = ct.bid(0)
        for kb in range(KB):
            xt = ct.load(x, index=(m, kb), shape=(QTM, B))
            amax = ct.max(ct.maximum(xt, 0.0 - xt), axis=-1, keepdims=True)
            ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
            byte = ct.where(ct.greater_equal(ea, OFF), ea - OFF, ct.full((QTM, 1), 0, ct.uint32))
            sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
            fp8 = ct.bitcast(ct.astype(xt / ct.broadcast_to(sc, (QTM, B)), elem), ct.uint8)
            ct.store(fp8_out, index=(m, kb), tile=fp8)
            ct.store(scale_out, index=(m, kb), tile=ct.astype(byte, ct.uint8))
    return _q


def _quant_col(elem):     # B(K,N) f32 → fp8(K,N) + e8m0(K//32,N); block of 32 along K (axis 0)
    @ct.kernel
    def _q(w, fp8_out, scale_out):
        kb, nb = ct.bid(0), ct.bid(1)
        wt = ct.load(w, index=(kb, nb), shape=(B, QWN))
        amax = ct.max(ct.maximum(wt, 0.0 - wt), axis=0, keepdims=True)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
        byte = ct.where(ct.greater_equal(ea, OFF), ea - OFF, ct.full((1, QWN), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        fp8 = ct.bitcast(ct.astype(wt / ct.broadcast_to(sc, (B, QWN)), elem), ct.uint8)
        ct.store(fp8_out, index=(kb, nb), tile=fp8)
        ct.store(scale_out, index=(kb, nb), tile=ct.astype(byte, ct.uint8))
    return _q


def _mm(elem):
    @ct.kernel(occupancy=2)
    def k(a, asc, b, bsc, out, KB: ct.Constant[int],
          TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
        m, n = ct.bid(0), ct.bid(1)
        acc = ct.zeros((TM, TN), ct.float32); KS = TK // B
        for kk in range(KB):
            ta  = ct.bitcast(ct.load(a, index=(m, kk), shape=(TM, TK)), elem)
            tas = ct.bitcast(ct.load(asc, index=(m, kk), shape=(TM, KS)), ct.float8_e8m0fnu)
            tb  = ct.bitcast(ct.load(b, index=(kk, n), shape=(TK, TN)), elem)
            tbs = ct.bitcast(ct.load(bsc, index=(kk, n), shape=(KS, TN)), ct.float8_e8m0fnu)
            acc = ct.mma_scaled(ta, tas, tb, tbs, acc)
        ct.store(out, index=(m, n), tile=acc)
    return k


def run(name, A, Bm):
    M, K = A.shape; _, N = Bm.shape
    gA, gB = _GpuArray(A), _GpuArray(Bm)
    ref = A.astype(np.float64) @ Bm.astype(np.float64); rn = np.abs(ref).max() + 1e-30
    print(f"  {name}:")
    res = {}
    for tag, elem in (("E4M3", ct.float8_e4m3fn), ("E5M2", ct.float8_e5m2)):
        try:
            Af = _GpuArray(np.zeros((M, K), np.uint8)); As = _GpuArray(np.zeros((M, K // B), np.uint8))
            Bf = _GpuArray(np.zeros((K, N), np.uint8)); Bs = _GpuArray(np.zeros((K // B, N), np.uint8))
            C = _GpuArray(np.zeros((M, N), np.float32))
            ct.launch(si, (M // QTM, 1, 1), _quant_row(elem), (gA, Af, As, K // B))
            ct.launch(si, (K // B, N // QWN, 1), _quant_col(elem), (gB, Bf, Bs))
            ct.launch(si, (M // 128, N // 128, 1), _mm(elem), (Af, As, Bf, Bs, C, K // 128, 128, 128, 128))
            cudart.cudaStreamSynchronize(si)
            err = float(np.abs(C.to_numpy().astype(np.float64) - ref).max() / rn)
            res[tag] = err; print(f"    {tag}: OK   rel-err {err:.3%}")
        except Exception as e:
            res[tag] = None; print(f"    {tag}: FAIL  {type(e).__name__}: {str(e)[:90]}")
    if res.get("E4M3") and res.get("E5M2"):
        win = "E4M3" if res["E4M3"] < res["E5M2"] else "E5M2"
        print(f"    → lower error: {win}  ({res['E4M3']/res['E5M2']:.2f}× E4M3/E5M2)")
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = N = 128; K = 256
    mag = 10.0 ** rng.uniform(-3, 1, size=(M, 1))                      # per-row wide cross-block range
    A1 = (rng.standard_normal((M, K)) * mag).astype(np.float32)
    Bm = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    run("smooth gradient (per-row wide range)", A1, Bm)
    A2 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    A2[:, ::B] *= 50.0                                                 # one 50x element per 32-block
    run("within-block outliers (one 50x per block)", A2, Bm)
