"""Measure the REAL achievable tensor-core peak on THIS GPU (RTX 5080 Laptop sm_120a) with a
pure-compute MMA microkernel: operands generated in-register (ct.full), a long in-kernel MMA
loop, NO HBM traffic. This is the true ceiling MFU should be measured against — the bench's
PEAK_BF16=165 was an assumption; the 5080 desktop is ~112 TFLOPS BF16 and the Laptop lower.

If our layer GEMMs (BF16 67 / MXFP8 139-162) are near THIS measured peak, we're not leaving
perf on the table — the cuda-tile GEMM is as good as the hardware gives (Blackwell sm_120
cuBLASLt itself regresses on large GEMM)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])

class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.nb // np.dtype(np.float32).itemsize, np.float32); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

TM, TN, TK = 64, 64, 64
ITERS = 512   # MMA iterations per block (compute-bound, no memory)
# 4 INDEPENDENT accumulator chains → ILP hides MMA latency → true tensor-core peak
# (a single serial acc=mma(a,b,acc) chain is latency-bound, undercounts).

@ct.kernel(occupancy=2)
def peak_bf16(out, NIT: ct.Constant[int]):
    a0 = ct.full((TM, TK), 1.0, ct.bfloat16); b0 = ct.full((TK, TN), 1.0, ct.bfloat16)
    c0 = ct.zeros((TM, TN), ct.float32); c1 = ct.zeros((TM, TN), ct.float32)
    c2 = ct.zeros((TM, TN), ct.float32); c3 = ct.zeros((TM, TN), ct.float32)
    for _ in range(NIT):
        c0 = ct.mma(a0, b0, c0); c1 = ct.mma(a0, b0, c1)
        c2 = ct.mma(a0, b0, c2); c3 = ct.mma(a0, b0, c3)
    ct.store(out, index=(ct.bid(0), 0), tile=c0 + c1 + c2 + c3)

@ct.kernel(occupancy=2)
def peak_mxfp8(out, NIT: ct.Constant[int]):
    a0 = ct.full((TM, TK), 1.0, ct.float8_e4m3fn); b0 = ct.full((TK, TN), 1.0, ct.float8_e4m3fn)
    asc = ct.full((TM, TK // 32), 1.0, ct.float8_e8m0fnu); bsc = ct.full((TK // 32, TN), 1.0, ct.float8_e8m0fnu)
    c0 = ct.zeros((TM, TN), ct.float32); c1 = ct.zeros((TM, TN), ct.float32)
    c2 = ct.zeros((TM, TN), ct.float32); c3 = ct.zeros((TM, TN), ct.float32)
    for _ in range(NIT):
        c0 = ct.mma_scaled(a0, asc, b0, bsc, c0); c1 = ct.mma_scaled(a0, asc, b0, bsc, c1)
        c2 = ct.mma_scaled(a0, asc, b0, bsc, c2); c3 = ct.mma_scaled(a0, asc, b0, bsc, c3)
    ct.store(out, index=(ct.bid(0), 0), tile=c0 + c1 + c2 + c3)

@ct.kernel(occupancy=2)
def peak_mxfp4(out, NIT: ct.Constant[int]):
    a0 = ct.full((TM, TK), 1.0, ct.float4_e2m1fn); b0 = ct.full((TK, TN), 1.0, ct.float4_e2m1fn)
    asc = ct.full((TM, TK // 32), 1.0, ct.float8_e8m0fnu); bsc = ct.full((TK // 32, TN), 1.0, ct.float8_e8m0fnu)
    c0 = ct.zeros((TM, TN), ct.float32); c1 = ct.zeros((TM, TN), ct.float32)
    c2 = ct.zeros((TM, TN), ct.float32); c3 = ct.zeros((TM, TN), ct.float32)
    for _ in range(NIT):
        c0 = ct.mma_scaled(a0, asc, b0, bsc, c0); c1 = ct.mma_scaled(a0, asc, b0, bsc, c1)
        c2 = ct.mma_scaled(a0, asc, b0, bsc, c2); c3 = ct.mma_scaled(a0, asc, b0, bsc, c3)
    ct.store(out, index=(ct.bid(0), 0), tile=c0 + c1 + c2 + c3)

def tmms(kern, grid, args, it=30, wm=8):
    for _ in range(wm): ct.launch(SI, grid, kern, args)
    cudart.cudaStreamSynchronize(SI); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, SI)
    for _ in range(it): ct.launch(SI, grid, kern, args)
    cudart.cudaEventRecord(t1, SI); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / it

if __name__ == "__main__":
    props = cudart.cudaGetDeviceProperties(0)[1]
    nsm = props.multiProcessorCount
    print(f"GPU: {props.name.decode()}  SMs={nsm}"); print("=" * 64)
    GRID = (nsm * 4, 1, 1)   # 4 blocks/SM, fill the GPU
    out = GA(np.zeros((GRID[0] * TM, TN), np.float32))
    flop = 2.0 * TM * TN * TK * ITERS * 4 * GRID[0]   # ×4 accumulator chains
    for nm, k in [("BF16  (mma)", peak_bf16), ("MXFP8 (mma_scaled)", peak_mxfp8), ("MXFP4 (mma_scaled)", peak_mxfp4)]:
        try:
            t = tmms(k, GRID, (out, ITERS))
            print(f"  {nm:20s}: {flop/(t/1e3)/1e12:6.0f} TFLOPS  (pure-compute peak)")
        except Exception as e:
            print(f"  {nm:20s}: FAIL {str(e)[:60]}")
    print("=" * 64)
    print("  → MFU should be measured vs THESE, not the 165/330 assumption.")
