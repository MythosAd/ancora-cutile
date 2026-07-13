"""THE megakernel-premise probe: can a persistent launch overlap a compute-bound GEMM with a
bandwidth-bound sweep at BLOCK granularity? (cuda-tile 1.5.0, static work-queue pattern.)

Independent ops (no data deps — the "same-layer independent operator group" case):
  GEMM : 2048^3 bf16 (512 tiles TM128/TN64, ~68 TF compute-bound, DRAM ~12%)
  sweep: f32 triad out = a*x + y, 1024 tiles (~100 MB traffic, BW-bound, ~0 FLOP)
Ideal complementary overlap → t_mixed ≈ max(t_gemm, t_sweep); no overlap → sum.
Variants: (a) two sequential launches (today's path), (b) ONE persistent launch, work list
INTERLEAVED 1 GEMM : 2 sweep (both types co-resident per SM), (c) persistent SEGREGATED
(all GEMM tiles then all sweep tiles — isolates persistent overhead from interleave gain).
Both outputs must stay BITWISE == the standard kernels (tile ownership unchanged)."""
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
G = 120
M = N = K = 2048
MB, NB, KB = M // TM, N // TN, K // TK
NT_G = MB * NB                       # 512 GEMM tiles
NT_S = 1024                          # sweep tiles (each (TM,TN) f32)
SB = NT_S                            # sweep arrays are (NT_S*TM, TN)
NT = NT_G + NT_S
TPB = (NT + G - 1) // G


@ct.kernel(occupancy=2)
def _gemm_std(A, B, C, KB_: ct.Constant[int]):
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(KB_):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM, TK), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK, TN), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _sweep_std(X, Y, O):
    i = ct.bid(0)
    x = ct.load(X, index=(i, 0), shape=(TM, TN), latency=10)
    y = ct.load(Y, index=(i, 0), shape=(TM, TN), latency=10)
    ct.store(O, index=(i, 0), tile=1.0009765625 * x + y)


def _gemm_tile(A, B, C, KB_, m, n):
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(KB_):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM, TK), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK, TN), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


def _sweep_tile(X, Y, O, i):
    x = ct.load(X, index=(i, 0), shape=(TM, TN), latency=10)
    y = ct.load(Y, index=(i, 0), shape=(TM, TN), latency=10)
    ct.store(O, index=(i, 0), tile=1.0009765625 * x + y)


@ct.kernel(occupancy=2)
def _mixed_interleaved(A, B, C, X, Y, O, KB_: ct.Constant[int]):
    """work list = 1 GEMM tile : 2 sweep tiles (512 + 1024). w%3==0 → GEMM w//3;
    else sweep 2*(w//3) + (w%3-1). Both op types co-resident on every SM."""
    b = ct.bid(0)
    for it in range(TPB):
        w = it * G + b
        if w < NT:
            g = w // 3
            r = w % 3
            if r == 0:
                _gemm_tile(A, B, C, KB_, g // NB, g % NB)
            else:
                _sweep_tile(X, Y, O, 2 * g + (r - 1))


@ct.kernel(occupancy=2)
def _mixed_segregated(A, B, C, X, Y, O, KB_: ct.Constant[int]):
    """Same persistent launch, but all GEMM tiles first, then all sweep tiles (control)."""
    b = ct.bid(0)
    for it in range(TPB):
        w = it * G + b
        if w < NT:
            if w < NT_G:
                _gemm_tile(A, B, C, KB_, w // NB, w % NB)
            else:
                _sweep_tile(X, Y, O, w - NT_G)


def tim(fn, reps=30):
    fn(); sync()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps): fn()
        sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
    return best


def main():
    rng = np.random.default_rng(0)
    A = _GpuArray(_f32bf(rng.standard_normal((M, K)).astype(np.float32)))
    B = _GpuArray(_f32bf(rng.standard_normal((K, N)).astype(np.float32)))
    Cs, Cm = _GpuArray.zeros((M, N), np.uint16), _GpuArray.zeros((M, N), np.uint16)
    X = _GpuArray(rng.standard_normal((SB * TM, TN)).astype(np.float32))
    Yv = _GpuArray(rng.standard_normal((SB * TM, TN)).astype(np.float32))
    Os, Om = _GpuArray.zeros((SB * TM, TN), np.float32), _GpuArray.zeros((SB * TM, TN), np.float32)

    ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB))
    ct.launch(si, (NT_S,), _sweep_std, (X, Yv, Os))
    ct.launch(si, (G,), _mixed_interleaved, (A, B, Cm, X, Yv, Om, KB)); sync()
    d_c = int((Cm.to_numpy() != Cs.to_numpy()).sum())
    d_o = int((Om.to_numpy() != Os.to_numpy()).sum())

    t_g = tim(lambda: ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB)))
    t_s = tim(lambda: ct.launch(si, (NT_S,), _sweep_std, (X, Yv, Os)))
    t_seq = tim(lambda: (ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB)),
                         ct.launch(si, (NT_S,), _sweep_std, (X, Yv, Os))))
    t_mix = tim(lambda: ct.launch(si, (G,), _mixed_interleaved, (A, B, Cm, X, Yv, Om, KB)))
    t_seg = tim(lambda: ct.launch(si, (G,), _mixed_segregated, (A, B, Cm, X, Yv, Om, KB)))

    # (d) TWO STREAMS — the only other in-DSL overlap mechanism (independent buffers → safe)
    so2 = dev.create_stream(); si2 = int(so2.__cuda_stream__()[1])
    def two_stream():
        ct.launch(si, (MB, NB), _gemm_std, (A, B, Cs, KB))
        ct.launch(si2, (NT_S,), _sweep_std, (X, Yv, Os))
    def sync2(): sync(); cudart.cudaStreamSynchronize(si2)
    two_stream(); sync2()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(30): two_stream()
        sync2(); best = min(best, (time.perf_counter() - t) / 30 * 1e6)
    t_2s = best

    ideal = max(t_g, t_s)
    print(f"  gemm-only {t_g:6.1f} us | sweep-only {t_s:6.1f} us | ideal-overlap max() = {ideal:6.1f} us")
    print(f"  (a) sequential launches : {t_seq:6.1f} us")
    print(f"  (b) persistent INTERLEAVED: {t_mix:6.1f} us  ({t_seq/t_mix:.2f}x vs seq; "
          f"captures {max(0.0, (t_seq-t_mix))/(t_seq-ideal)*100:.0f}% of the overlap window)")
    print(f"  (c) persistent SEGREGATED : {t_seg:6.1f} us  ({t_seq/t_seg:.2f}x vs seq)")
    print(f"  (d) two streams           : {t_2s:6.1f} us  ({t_seq/t_2s:.2f}x vs seq; "
          f"captures {max(0.0, (t_seq-t_2s))/(t_seq-ideal)*100:.0f}% of the overlap window)")
    print(f"  bitwise: C diffs {d_c}, O diffs {d_o}  {'OK' if d_c == 0 and d_o == 0 else 'FAIL'}")
    for g in (A, B, Cs, Cm, X, Yv, Os, Om): g.free()
    return d_c == 0 and d_o == 0


if __name__ == "__main__":
    print(f"heterogeneous persistent batch: GEMM + BW-sweep in ONE launch (cuda-tile {ct.__version__})")
    print("=" * 88)
    ok = main()
    print("  PASS" if ok else "  FAIL")
