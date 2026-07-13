"""Persistent work-queue GEMM probe (cuda-tile 1.5.0: atomics + num_blocks + runtime guards).

The megakernel seed: instead of grid == tile-count, launch a FIXED grid (~2 blocks/SM) and
let each block loop over output tiles — the pattern a persistent multi-op scheduler needs.
Two schedulers, both BITWISE-safe (each output tile owned by ONE block, full-K in-block,
scheduling order never touches bits):
  static : tid = it*G + bid   (deterministic round-robin, no atomics)
  atomic : tid = atomic_add(counter, 1)  (dynamic load balancing — what heterogeneous
           work in a real megakernel would use; int tickets are order-free)
Checks: bitwise vs the standard grid-per-tile GEMM + wall time at a fat and a thin shape."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)

TM, TN, TK = 128, 64, 64
G = 120                       # persistent grid: 60 SMs x occupancy 2


@ct.kernel(occupancy=2)
def _gemm_std(A, B, C, KB: ct.Constant[int]):
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM, TK), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK, TN), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _gemm_pers_static(A, B, C, KB: ct.Constant[int], NB: ct.Constant[int],
                      NT: ct.Constant[int], TPB: ct.Constant[int]):
    """Fixed grid (G,); block b computes tiles b, b+G, b+2G, ... (deterministic)."""
    b = ct.bid(0)
    for it in range(TPB):
        tid = it * G + b
        if tid < NT:
            m = tid // NB
            n = tid % NB
            acc = ct.zeros((TM, TN), ct.float32)
            for k in range(KB):
                ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM, TK), latency=10), ct.bfloat16)
                tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK, TN), latency=10), ct.bfloat16)
                acc = ct.mma(ta, tb, acc)
            ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _gemm_pers_atomic(A, B, C, cnt, KB: ct.Constant[int], NB: ct.Constant[int],
                      NT: ct.Constant[int], TPB: ct.Constant[int]):
    """Fixed grid (G,); each block pulls work tickets from a device counter."""
    for it in range(TPB):
        t = ct.atomic_add(cnt, 0, 1)
        tid = ct.reshape(t, ())
        if tid < NT:
            m = tid // NB
            n = tid % NB
            acc = ct.zeros((TM, TN), ct.float32)
            for k in range(KB):
                ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM, TK), latency=10), ct.bfloat16)
                tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK, TN), latency=10), ct.bfloat16)
                acc = ct.mma(ta, tb, acc)
            ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _reset(cnt):
    ct.store(cnt, index=0, tile=ct.zeros((1,), ct.int32))


def bench(M, N, K):
    MB, NB, KB = M // TM, N // TN, K // TK
    NT = MB * NB; TPB = (NT + G - 1) // G
    rng = np.random.default_rng(0)
    A = _GpuArray(_f32bf(rng.standard_normal((M, K)).astype(np.float32)))
    B = _GpuArray(_f32bf(rng.standard_normal((K, N)).astype(np.float32)))
    Cs, Cp, Ca = (_GpuArray.zeros((M, N), np.uint16) for _ in range(3))
    cnt = _GpuArray(np.zeros(1, np.int32))

    ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB))
    ct.launch(si, (G,), _gemm_pers_static, (A, B, Cp, KB, NB, NT, TPB))
    ct.launch(si, (1,), _reset, (cnt,))
    ct.launch(si, (G,), _gemm_pers_atomic, (A, B, Ca, cnt, KB, NB, NT, TPB)); sync()
    d_s = int((Cp.to_numpy() != Cs.to_numpy()).sum()); d_a = int((Ca.to_numpy() != Cs.to_numpy()).sum())

    def tim(fn, reps=40):
        fn(); sync()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps): fn()
            sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
        return best

    t_std = tim(lambda: ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB)))
    t_per = tim(lambda: ct.launch(si, (G,), _gemm_pers_static, (A, B, Cp, KB, NB, NT, TPB)))
    t_atm = tim(lambda: (ct.launch(si, (1,), _reset, (cnt,)),
                         ct.launch(si, (G,), _gemm_pers_atomic, (A, B, Ca, cnt, KB, NB, NT, TPB))))
    tf = 2 * M * N * K / 1e6
    print(f"  {M}x{N}x{K} (NT={NT:4d}, TPB={TPB}): std {t_std:6.1f} us ({tf/t_std:5.1f} TF)"
          f" | pers-static {t_per:6.1f} ({tf/t_per:5.1f} TF, {t_std/t_per:.2f}x, bits {'OK' if d_s==0 else d_s})"
          f" | pers-atomic {t_atm:6.1f} ({tf/t_atm:5.1f} TF, {t_std/t_atm:.2f}x, bits {'OK' if d_a==0 else d_a})")
    for g in (A, B, Cs, Cp, Ca, cnt): g.free()
    return d_s == 0 and d_a == 0


if __name__ == "__main__":
    print(f"persistent work-queue GEMM (cuda-tile {ct.__version__}, G={G} blocks fixed)")
    print("=" * 100)
    ok = bench(2048, 2048, 2048)          # fat: NT=512 tiles >> G
    ok &= bench(2048, 1024, 1024)         # layer proj shape: NT=256
    ok &= bench(512, 1024, 1024)          # small M: NT=64 < G (underfill regime)
    print("  " + ("PASS (both schedulers bitwise == standard)" if ok else "FAIL"))
