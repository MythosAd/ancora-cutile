"""Sub-probe for the E4M3 dgrad: dx = dy @ Wᵀ in FP8.  The contraction is N; both dy(M,N) and
W(K,N) are block-scaled per-32 along N (axis 1). For the mma we need b = Wᵀ (N,K) and b_scale
transposed too. Question: does ct.mma_scaled accept ct.transpose() of an fp8 operand AND its
e8m0 scale (Option A, in-kernel transpose)? If yes, we avoid maintaining a second pre-transposed
weight quant (Option B). Validate vs the fp64 reference dx = dy @ Wᵀ."""
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

B = 32; QTM = 64; OFF = 7
E4 = ct.float8_e4m3fn


@ct.kernel                                        # block of 32 along axis 1 (the N / contraction dim)
def _q_row(x, fp8_out, scale_out, KB: ct.Constant[int]):
    m = ct.bid(0)
    for kb in range(KB):
        xt = ct.load(x, index=(m, kb), shape=(QTM, B))
        amax = ct.max(ct.maximum(xt, 0.0 - xt), axis=-1, keepdims=True)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
        byte = ct.where(ct.greater_equal(ea, OFF), ea - OFF, ct.full((QTM, 1), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        ct.store(fp8_out, index=(m, kb), tile=ct.bitcast(ct.astype(xt / ct.broadcast_to(sc, (QTM, B)), E4), ct.uint8))
        ct.store(scale_out, index=(m, kb), tile=ct.astype(byte, ct.uint8))


@ct.kernel(occupancy=2)                           # dx = dy @ Wᵀ ; reduce over N ; W transposed IN-KERNEL
def _dx_fp8_T(dy, dys, W, Ws, dx, NB: ct.Constant[int],
              TM: ct.Constant[int], TK: ct.Constant[int], TN: ct.Constant[int]):
    m, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TK), ct.float32); KS = TN // B
    for n in range(NB):
        a   = ct.bitcast(ct.load(dy,  index=(m, n), shape=(TM, TN)), E4)
        asc = ct.bitcast(ct.load(dys, index=(m, n), shape=(TM, KS)), ct.float8_e8m0fnu)
        w   = ct.bitcast(ct.load(W,   index=(k, n), shape=(TK, TN)), E4)
        ws  = ct.bitcast(ct.load(Ws,  index=(k, n), shape=(TK, KS)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(a, asc, ct.transpose(w), ct.transpose(ws), acc)   # b=Wᵀ, b_scale=Wsᵀ
    ct.store(dx, index=(m, k), tile=acc)


def main():
    rng = np.random.default_rng(0)
    M, K, N = 128, 128, 256
    mag = 10.0 ** rng.uniform(-3, 1, size=(M, 1))
    dy = (rng.standard_normal((M, N)) * mag).astype(np.float32)
    W = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    ref = dy.astype(np.float64) @ W.T.astype(np.float64); rn = np.abs(ref).max() + 1e-30
    gdy, gW = _GpuArray(dy), _GpuArray(W)
    dyf, dys = _GpuArray(np.zeros((M, N), np.uint8)), _GpuArray(np.zeros((M, N // B), np.uint8))
    Wf, Ws = _GpuArray(np.zeros((K, N), np.uint8)), _GpuArray(np.zeros((K, N // B), np.uint8))
    dx = _GpuArray(np.zeros((M, K), np.float32))
    try:
        ct.launch(si, (M // QTM, 1, 1), _q_row, (gdy, dyf, dys, N // B))
        ct.launch(si, (K // QTM, 1, 1), _q_row, (gW, Wf, Ws, N // B))
        ct.launch(si, (M // 128, K // 128, 1), _dx_fp8_T, (dyf, dys, Wf, Ws, dx, N // 128, 128, 128, 128))
        cudart.cudaStreamSynchronize(si)
        err = float(np.abs(dx.to_numpy().astype(np.float64) - ref).max() / rn)
        ok = err < 0.08
        print(f"  Option A (in-kernel transpose of fp8 operand + e8m0 scale): {'OK' if ok else 'WRONG'}  "
              f"rel-err {err:.3%}  {'→ use Option A (simplest)' if ok else '→ fall back to Option B'}")
    except Exception as e:
        print(f"  Option A: FAIL  {type(e).__name__}: {str(e)[:110]}")
        print("  → mma_scaled rejects transposed operands → use Option B (pre-transposed weight quant)")


if __name__ == "__main__":
    main()
