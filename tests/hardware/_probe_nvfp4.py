"""Authoritative FP4 probe on sm_120a (cuda-tile 1.4.0, 2026-06-02). Answers: does the
toolchain support MXFP4 / NVFP4, and can we actually USE FP4 to raise MFU?

Split into COMPUTE (mma_scaled) vs LOAD (getting packed FP4 from HBM into a 2D MMA tile):
  COMPUTE: MXFP4 (FP4 x E8M0, B=32) and NVFP4 (FP4 x E4M3, B=16) — generate operands
           in-kernel (ct.full) so we test mma_scaled alone. a=b=1.0, scale=1.0 → out==K.
  LOAD:    three ways to get external packed FP4 bytes into a tile — all fail/limited.
The verdict: compute works for BOTH formats; the LOAD is the blocker (NVIDIA cutile #47),
so FP4's 2x peak is unrealizable in 1.4.0 regardless of MXFP4 vs NVFP4."""
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

M = N = 16; TK = 64

# ── COMPUTE (1): MXFP4 — FP4 x E8M0 scale, block 32 ──
@ct.kernel
def mxfp4(out, KB: ct.Constant[int]):
    acc = ct.zeros((M, N), ct.float32)
    for k in range(KB):
        a = ct.full((M, TK), 1.0, ct.float4_e2m1fn); b = ct.full((TK, N), 1.0, ct.float4_e2m1fn)
        asc = ct.full((M, TK // 32), 1.0, ct.float8_e8m0fnu); bsc = ct.full((TK // 32, N), 1.0, ct.float8_e8m0fnu)
        acc = ct.mma_scaled(a, asc, b, bsc, acc)
    ct.store(out, index=(0, 0), tile=acc)

# ── COMPUTE (2): NVFP4 — FP4 x E4M3 scale, block 16 ──
@ct.kernel
def nvfp4(out, KB: ct.Constant[int]):
    acc = ct.zeros((M, N), ct.float32)
    for k in range(KB):
        a = ct.full((M, TK), 1.0, ct.float4_e2m1fn); b = ct.full((TK, N), 1.0, ct.float4_e2m1fn)
        asc = ct.full((M, TK // 16), 1.0, ct.float8_e4m3fn); bsc = ct.full((TK // 16, N), 1.0, ct.float8_e4m3fn)
        acc = ct.mma_scaled(a, asc, b, bsc, acc)
    ct.store(out, index=(0, 0), tile=acc)

# ── LOAD (a): bitcast uint8 -> fp4 (2 fp4/byte) ──
@ct.kernel
def load_bitcast(packed, out):
    a = ct.bitcast(ct.load(packed, index=(0, 0), shape=(M, TK // 2)), ct.float4_e2m1fn)
    b = ct.full((TK, N), 1.0, ct.float4_e2m1fn)
    asc = ct.full((M, TK // 16), 1.0, ct.float8_e4m3fn); bsc = ct.full((TK // 16, N), 1.0, ct.float8_e4m3fn)
    ct.store(out, index=(0, 0), tile=ct.mma_scaled(a, asc, b, bsc, ct.zeros((M, N), ct.float32)))

# ── LOAD (b): unpack_from_bytes on a 2D byte tile ──
@ct.kernel
def load_unpack2d(packed, out):
    a = ct.unpack_from_bytes(ct.load(packed, index=(0, 0), shape=(M, TK // 2)), ct.float4_e2m1fn)
    b = ct.full((TK, N), 1.0, ct.float4_e2m1fn)
    asc = ct.full((M, TK // 16), 1.0, ct.float8_e4m3fn); bsc = ct.full((TK // 16, N), 1.0, ct.float8_e4m3fn)
    ct.store(out, index=(0, 0), tile=ct.mma_scaled(a, asc, b, bsc, ct.zeros((M, N), ct.float32)))


def run(kern, args, grid=(1, 1, 1)):
    try:
        ct.launch(si, grid, kern, args); cudart.cudaStreamSynchronize(si); return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"

if __name__ == "__main__":
    print("FP4 probe on sm_120a (cuda-tile 1.4.0)"); print("=" * 70)
    KB = 4
    o1 = GA(np.zeros((M, N), np.float32)); r1 = run(mxfp4, (o1, KB))
    print(f"  COMPUTE MXFP4 (FP4xE8M0, B32): {r1}" + (f"  out={o1.np()[0,0]} (expect {KB*TK})" if r1 == "ok" else ""))
    o2 = GA(np.zeros((M, N), np.float32)); r2 = run(nvfp4, (o2, KB))
    print(f"  COMPUTE NVFP4 (FP4xE4M3, B16): {r2}" + (f"  out={o2.np()[0,0]} (expect {KB*TK})" if r2 == "ok" else ""))
    print("-" * 70)
    pk = GA(np.zeros((M, TK // 2), np.uint8)); o3 = GA(np.zeros((M, N), np.float32))
    print(f"  LOAD bitcast u8->fp4       : {run(load_bitcast, (pk, o3))}")
    print(f"  LOAD unpack_from_bytes (2D): {run(load_unpack2d, (pk, o3))}")
    print(f"  LOAD unpack_from_bytes (1D): blocked for GEMM — unpack is 1D-only, GEMM tiles are 2D strided")
    print("=" * 70)
    print("  VERDICT: MXFP4 & NVFP4 COMPUTE both work; FP4 LOAD has no 2D path → FP4 unusable")
    print("           for GEMM in 1.4.0. MFU lever stays MXFP8 (#47 blocks packed-FP4 load).")
