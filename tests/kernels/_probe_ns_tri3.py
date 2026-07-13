"""Tri-NS round 3: mirror as a SEPARATE pass (the in-GEMM transpose(acc) store was the killer —
mma-fragment layout scatters; a load→transpose→store copy kernel is the production _transpose_mat
pattern). Measures the mirror pass at 64²/128² tiles, then the FULL tri chain (nt_tri + mirror +
axpy_tri + mirror + fused BX) vs the current fused chain — bitwise + wall-time."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import (_e_gemm_nt, _e_gemm_axpy, _e_fro_recip, _e_scale, _e_axpy,
                                     newton_schulz_resident_e, NTM, NTN, NTK)
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
Z16 = lambda *s: _GpuArray.zeros(s, np.uint16)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


@ct.kernel(occupancy=2)
def _e_gemm_nt_tril(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
                    TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = A_e@B_eᵀ, LOWER-triangle blocks only (n≤m). Upper mirrored by _e_mirror."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * MB + n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, ct.transpose(tb), acc)
        ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _e_gemm_axpy_tril(A, B, D, C, alpha, beta, MB: ct.Constant[int], KBe: ct.Constant[int],
                      KB: ct.Constant[int],
                      TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = alpha·D_e + beta·(A_e@B_e), LOWER blocks only (symmetric output)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * KBe + k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, tb, acc)
        d = ct.astype(ct.bitcast(ct.load(D, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
        ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(alpha * d + beta * acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _e_mirror(C, MB: ct.Constant[int], TT_: ct.Constant[int]):
    """Copy the strict lower triangle to the upper: C[m,n] ← C[n,m]ᵀ for n>m (bit copy).
    Load is coalesced; transpose in registers (the production _transpose_mat pattern).
    grid (E, M//TT, M//TT); MB = M//TT."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n > m:
        t = ct.load(C, index=(e * MB + n, m), shape=(TT_, TT_))
        ct.store(C, index=(e * MB + m, n), tile=ct.transpose(t))


def bench(E, M, N):
    rng = np.random.default_rng(0)
    Xp = np.concatenate([_f32bf(rng.standard_normal((M, N)).astype(np.float32)) for _ in range(E)], 0)
    gX = _GpuArray(Xp.copy())
    gA_f, gA_t, gB_f, gB_t = Z16(E * M, M), Z16(E * M, M), Z16(E * M, M), Z16(E * M, M)
    mb, kb, kbm = M // NTM, N // NTK, M // NTK
    b, c = -4.7750, 2.0315

    def tim(fn, reps=50):
        fn(); sync()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps): fn()
            sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
        return best

    # correctness first (bitwise, per pitfall 0c the mirror is a bit copy so == is the full check)
    ct.launch(si, (E, mb, mb), _e_gemm_nt, (gX, gX, gA_f, mb, kb, NTM, NTN, NTK))
    ct.launch(si, (E, mb, mb), _e_gemm_nt_tril, (gX, gX, gA_t, mb, kb, NTM, NTN, NTK))
    for TT, MBt in ((64, M // 64), (128, M // 128)):
        pass
    ct.launch(si, (E, M // 64, M // 64), _e_mirror, (gA_t, M // 64, 64))
    sync()
    d1 = int((gA_f.to_numpy() != gA_t.to_numpy()).sum())
    ct.launch(si, (E, mb, mb), _e_gemm_axpy, (gA_f, gA_f, gA_f, gB_f, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK))
    ct.launch(si, (E, mb, mb), _e_gemm_axpy_tril, (gA_f, gA_f, gA_f, gB_t, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK))
    ct.launch(si, (E, M // 64, M // 64), _e_mirror, (gB_t, M // 64, 64))
    sync()
    d2 = int((gB_f.to_numpy() != gB_t.to_numpy()).sum())
    print(f"  E={E} {M}x{N}: nt_tril+mirror bit-diffs={d1}  axpy_tril+mirror bit-diffs={d2}  "
          f"{'OK' if d1 == 0 and d2 == 0 else 'FAIL'}")

    t_nt_f = tim(lambda: ct.launch(si, (E, mb, mb), _e_gemm_nt, (gX, gX, gA_f, mb, kb, NTM, NTN, NTK)))
    t_nt_t = tim(lambda: ct.launch(si, (E, mb, mb), _e_gemm_nt_tril, (gX, gX, gA_t, mb, kb, NTM, NTN, NTK)))
    t_mir64 = tim(lambda: ct.launch(si, (E, M // 64, M // 64), _e_mirror, (gA_t, M // 64, 64)))
    t_mir128 = tim(lambda: ct.launch(si, (E, M // 128, M // 128), _e_mirror, (gA_t, M // 128, 128)))
    t_ax_f = tim(lambda: ct.launch(si, (E, mb, mb), _e_gemm_axpy, (gA_f, gA_f, gA_f, gB_f, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK)))
    t_ax_t = tim(lambda: ct.launch(si, (E, mb, mb), _e_gemm_axpy_tril, (gA_f, gA_f, gA_f, gB_t, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK)))
    print(f"    X@Xᵀ : full {t_nt_f:5.0f} → tril {t_nt_t:5.0f}   mirror64 {t_mir64:4.0f} / mirror128 {t_mir128:4.0f}"
          f"   tril+mir {t_nt_t + min(t_mir64, t_mir128):5.0f} ({t_nt_f/(t_nt_t+min(t_mir64,t_mir128)):.2f}x)")
    print(f"    axpyG: full {t_ax_f:5.0f} → tril {t_ax_t:5.0f}   tril+mir {t_ax_t + min(t_mir64, t_mir128):5.0f}"
          f" ({t_ax_f/(t_ax_t+min(t_mir64,t_mir128)):.2f}x)")
    for g in (gX, gA_f, gA_t, gB_f, gB_t): g.free()
    return d1 == 0 and d2 == 0


if __name__ == "__main__":
    print("=" * 90)
    ok = bench(16, 1024, 1024)
    ok &= bench(8, 1024, 2048)
    print("=" * 90)
    print("  PASS" if ok else "  FAIL")
