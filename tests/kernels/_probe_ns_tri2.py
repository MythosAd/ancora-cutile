"""Diagnose the tri X@Xᵀ slowdown: is the `if` guard a real branch (idle blocks retire) or
predicated (no savings)? Variants: full / tri / tri WITHOUT the mirror store (lower half only) /
tri without latency hint. Real shapes: square E=16 1024² (expert chain) + rect E=8 (1024,2048)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import _e_gemm_nt, NTM, NTN, NTK
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
Z16 = lambda *s: _GpuArray.zeros(s, np.uint16)


@ct.kernel(occupancy=2)
def _nt_tri(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
            TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * MB + n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, ct.transpose(tb), acc)
        cb = ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16)
        ct.store(C, index=(e * MB + m, n), tile=cb)
        if n < m:
            ct.store(C, index=(e * MB + n, m), tile=ct.transpose(cb))


@ct.kernel(occupancy=2)
def _nt_tri_nomirror(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
                     TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Lower blocks only, NO mirror store — isolates guard-skip savings from mirror-store cost."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * MB + n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, ct.transpose(tb), acc)
        ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _nt_tri_nolat(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
                  TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Guarded + mirror but WITHOUT the latency=10 prefetch hint (pipelining-vs-branch interaction)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_)), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * MB + n, k), shape=(TN_, TK_)), ct.bfloat16)
            acc = ct.mma(ta, ct.transpose(tb), acc)
        cb = ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16)
        ct.store(C, index=(e * MB + m, n), tile=cb)
        if n < m:
            ct.store(C, index=(e * MB + n, m), tile=ct.transpose(cb))


def bench(E, M, N):
    rng = np.random.default_rng(0)
    Xp = np.concatenate([_f32bf(rng.standard_normal((M, N)).astype(np.float32)) for _ in range(E)], 0)
    gX, gA = _GpuArray(Xp.copy()), Z16(E * M, M)
    mb, kb = M // NTM, N // NTK

    def tim(kern, reps=50):
        ct.launch(si, (E, mb, mb), kern, (gX, gX, gA, mb, kb, NTM, NTN, NTK)); sync()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps): ct.launch(si, (E, mb, mb), kern, (gX, gX, gA, mb, kb, NTM, NTN, NTK))
            sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
        return best

    t_full = tim(_e_gemm_nt)
    t_tri = tim(_nt_tri)
    t_nom = tim(_nt_tri_nomirror)
    t_nol = tim(_nt_tri_nolat)
    print(f"  E={E} {M}x{N} → ({M}²): full {t_full:5.0f}  tri {t_tri:5.0f} ({t_full/t_tri:.2f}x)  "
          f"tri-nomirror {t_nom:5.0f} ({t_full/t_nom:.2f}x)  tri-nolat {t_nol:5.0f} ({t_full/t_nol:.2f}x)")
    for g in (gX, gA): g.free()


if __name__ == "__main__":
    bench(16, 1024, 1024)     # the expert chain shape
    bench(8, 1024, 2048)      # the rect q/o group shape
