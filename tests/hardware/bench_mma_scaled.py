"""
ct.mma_scaled TFLOPS benchmark — RTX 5080 Laptop sm_120a
Measures actual FP8 MXFP8 tensor core throughput.

FLOPs counted as: 2 × M × N × K  (each fused multiply-add = 2 ops)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import grpo_rl.env
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

# ── setup ──────────────────────────────────────────────────────────────────
cudart.cudaFree(0)
dev = cc.Device(0)
dev.set_current()
stream_obj = dev.create_stream()
stream_int = int(stream_obj.__cuda_stream__()[1])

def alloc(shape, dtype):
    arr = np.ones(shape, dtype)
    err, ptr = cdrv.cuMemAlloc(arr.nbytes)
    cdrv.cuMemcpyHtoD(ptr, arr, arr.nbytes)
    class GA:
        def __init__(self, p, a):
            self._ptr = p
            self.__cuda_array_interface__ = {
                "shape": a.shape, "typestr": a.dtype.str,
                "data": (int(p), False), "version": 3,
            }
        def free(self): cdrv.cuMemFree(self._ptr)
    return GA(ptr, arr)

def cuda_event():
    err, e = cudart.cudaEventCreate()
    return e

def elapsed_ms(start, stop):
    cudart.cudaEventSynchronize(stop)
    err, ms = cudart.cudaEventElapsedTime(start, stop)
    return ms

# ── kernel ─────────────────────────────────────────────────────────────────
# Pure-compute kernel: uses ct.ones/ct.full to fill tiles, no global memory load.
# This measures PEAK tensor core throughput (upper bound; real kernels are memory-bound).
#
# Each block computes a TILE_M × TILE_N output tile by accumulating K_BLOCKS
# chunks of TILE_K=128 each, so effective K = K_BLOCKS × 128.
# FLOPs per block = 2 × TILE_M × TILE_N × (K_BLOCKS × 128)

TILE_M  = 128
TILE_N  = 128
TILE_K  = 128      # must be divisible by scale block size B=32
B       = 32       # MXFP8 scale block size
K_S     = TILE_K // B  # = 4 scale factors per tile

@ct.kernel
def mxfp8_bench(a, b, c,
                M: ct.Constant[int], N: ct.Constant[int],
                K_BLOCKS: ct.Constant[int]):
    """Load A/B from device memory (forces real HBM reads), scale=1.0 constant.
    This measures TC throughput bottlenecked by memory bandwidth for large K."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TILE_M, TILE_N), ct.float32)
    for k in range(K_BLOCKS):
        ta  = ct.astype(ct.load(a, index=(m, k), shape=(TILE_M, TILE_K)), ct.float8_e4m3fn)
        tb  = ct.astype(ct.load(b, index=(k, n), shape=(TILE_K, TILE_N)), ct.float8_e4m3fn)
        tsa = ct.full((TILE_M, K_S), 1.0, ct.float8_e8m0fnu)
        tsb = ct.full((K_S, TILE_N), 1.0, ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, tsa, tb, tsb, acc)
    ct.store(c, index=(m, n), tile=ct.astype(acc, ct.float32))

@ct.kernel
def mxfp4_bench(a, b, c,
                M: ct.Constant[int], N: ct.Constant[int],
                K_BLOCKS: ct.Constant[int]):
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TILE_M, TILE_N), ct.float32)
    for k in range(K_BLOCKS):
        ta  = ct.astype(ct.load(a, index=(m, k), shape=(TILE_M, TILE_K)), ct.float4_e2m1fn)
        tb  = ct.astype(ct.load(b, index=(k, n), shape=(TILE_K, TILE_N)), ct.float4_e2m1fn)
        tsa = ct.full((TILE_M, K_S), 1.0, ct.float8_e8m0fnu)
        tsb = ct.full((K_S, TILE_N), 1.0, ct.float8_e8m0fnu)
        acc = ct.mma_scaled(ta, tsa, tb, tsb, acc)
    ct.store(c, index=(m, n), tile=ct.astype(acc, ct.float32))

# ── benchmark runner ───────────────────────────────────────────────────────

def run_bench(name, kernel_fn, M, N, K_BLOCKS, warmup=10, iters=200, input_dtype=np.float32):
    """
    Returns TFLOPS.
    FLOPs = 2 × M × N × (K_BLOCKS × TILE_K) per kernel call.
    Loads from real device arrays to prevent compiler optimizing away work.
    """
    G_M = M  // TILE_M
    G_N = N  // TILE_N
    K_total = K_BLOCKS * TILE_K
    flops_per_call = 2 * M * N * K_total

    a = alloc((M,       K_total), input_dtype)
    b = alloc((K_total, N),      input_dtype)
    c = alloc((M,       N),      np.float32)

    def launch():
        ct.launch(stream_int, (G_M, G_N, 1), kernel_fn,
                  (a, b, c, M, N, K_BLOCKS))

    for _ in range(warmup):
        launch()
    stream_obj.sync()

    t0 = cuda_event()
    t1 = cuda_event()
    cudart.cudaEventRecord(t0, stream_obj.__cuda_stream__()[1])
    for _ in range(iters):
        launch()
    cudart.cudaEventRecord(t1, stream_obj.__cuda_stream__()[1])
    ms = elapsed_ms(t0, t1)

    total_flops = flops_per_call * iters
    tflops = total_flops / (ms / 1000) / 1e12

    print(f"  {name:<22} M=N={M} K={K_total}  "
          f"{ms/iters:.3f} ms/call  {tflops:.1f} TFLOPS")
    for x in (a, b, c): x.free()
    return tflops

# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"cuda-tile {ct.__version__}  sm_120a  MXFP8/MXFP4 ct.mma_scaled benchmark")
    print("=" * 70)
    print()

    results = {}

    print("--- MXFP8: float32输入（内存瓶颈参考）---")
    for M, K_B in [(4096, 32), (8192, 32)]:
        t = run_bench("MXFP8[f32 input]", mxfp8_bench, M, M, K_B)
        results[f"MXFP8_f32 M={M} K={K_B*TILE_K}"] = t

    print()
    print("--- MXFP8: uint8输入（FP8原生存储，1 byte/元素）---")
    for M, K_B in [(4096, 32), (8192, 32)]:
        t = run_bench("MXFP8[u8 input]",  mxfp8_bench, M, M, K_B,
                      input_dtype=np.uint8)
        results[f"MXFP8_u8 M={M} K={K_B*TILE_K}"] = t

    print()
    print("=" * 70)
    t_f32 = max(v for k,v in results.items() if "f32" in k)
    t_u8  = max(v for k,v in results.items() if "u8"  in k)
    print(f"  MXFP8 float32 input (bandwidth×4 wasted): {t_f32:.1f} TFLOPS")
    print(f"  MXFP8 uint8  input  (native FP8 storage): {t_u8:.1f} TFLOPS")
    print(f"  Ratio: {t_u8/t_f32:.1f}x  (should be ~4x if purely BW-bound)")
    print()
    fp8_peak_est = 1600
    print(f"  MFU (native FP8, vs ~{fp8_peak_est} TFLOPS est): {t_u8/fp8_peak_est*100:.1f}%")
    print("  (no SMEM tiling, no TMA — this is the unoptimized baseline)")
