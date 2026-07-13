"""Probe TRIANGULAR (symmetric-output) GEMM for the NS chain (task #32 kernel review).

X@Xᵀ and A@A (A symmetric) have SYMMETRIC outputs — a block (m,n) with n≤m can compute once and
mirror-store transpose(acc) to (n,m). The mirror is expected BIT-identical to computing (n,m)
directly (same products, same k order; IEEE multiply commutes bitwise; elementwise axpy commutes
with transpose). Needs: (a) a runtime `if n <= m` block guard in a cuda-tile kernel — compile+run
probe; (b) transpose-store of a 128×128 tile; (c) bitwise equality vs the full kernels, PER TILE
(pitfall 0c); (d) the wall-time win (~44% of those two GEMMs' blocks skipped)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import _e_gemm_nt, _e_gemm_axpy, NTM, NTN, NTK
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
Z16 = lambda *s: _GpuArray.zeros(s, np.uint16)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


# ── (a) minimal runtime-if probe ──────────────────────────────────────────────
@ct.kernel
def _tri_copy(X, Y, T: ct.Constant[int]):
    m, n = ct.bid(0), ct.bid(1)
    if n <= m:
        t = ct.load(X, index=(m, n), shape=(T, T))
        ct.store(Y, index=(m, n), tile=t)
        if n < m:                                    # nested guard: mirror the strict-lower tile
            ct.store(Y, index=(n, m), tile=ct.transpose(t))


# ── (b/c) triangular X@Xᵀ and fused axpy A@A with mirror store ───────────────
@ct.kernel(occupancy=2)
def _e_gemm_nt_tri(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
                   TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = A_e @ B_eᵀ with SYMMETRIC output: only n≤m blocks compute; (n,m) is the
    mirror transpose (bit-identical: same products, same k order). Requires TM==TN."""
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
def _e_gemm_axpy_tri(A, B, D, C, alpha, beta, MB: ct.Constant[int], KBe: ct.Constant[int],
                     KB: ct.Constant[int],
                     TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = alpha·D_e + beta·(A_e@B_e) with SYMMETRIC output (A,B,D symmetric — the NS
    B = b·A + c·A@A step). Only n≤m blocks compute; mirror-store the transpose. TM==TN."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(e * KBe + k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, tb, acc)
        d = ct.astype(ct.bitcast(ct.load(D, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
        cb = ct.bitcast(ct.astype(alpha * d + beta * acc, ct.bfloat16), ct.uint16)
        ct.store(C, index=(e * MB + m, n), tile=cb)
        if n < m:
            ct.store(C, index=(e * MB + n, m), tile=ct.transpose(cb))


def probe_if():
    M, T = 512, 128
    rng = np.random.default_rng(0)
    X = rng.integers(0, 60000, (M, M)).astype(np.uint16)
    gX, gY = _GpuArray(X), _GpuArray(np.zeros((M, M), np.uint16))
    ct.launch(si, (M // T, M // T, 1), _tri_copy, (gX, gY, T))
    sync()
    Y = gY.to_numpy()
    ref = np.zeros_like(X)
    for m in range(0, M, T):
        for n in range(0, M, T):
            if n <= m:
                ref[m:m+T, n:n+T] = X[m:m+T, n:n+T]
                if n < m:
                    ref[n:n+T, m:m+T] = X[m:m+T, n:n+T].T
    ok = (Y == ref).all()
    print(f"[a] runtime `if n<=m` guard + nested mirror + 128² transpose-store: {'OK' if ok else 'FAIL'}")
    for g in (gX, gY): g.free()
    return bool(ok)


def probe_tri_gemms(E=4, M=1024, N=2048):
    rng = np.random.default_rng(1)
    Xp = np.concatenate([_f32bf(rng.standard_normal((M, N)).astype(np.float32)) for _ in range(E)], 0)
    gX = _GpuArray(Xp.copy())
    gA_f, gA_t = Z16(E * M, M), Z16(E * M, M)
    mb, kb = M // NTM, N // NTK
    ct.launch(si, (E, mb, mb), _e_gemm_nt, (gX, gX, gA_f, mb, kb, NTM, NTN, NTK))
    ct.launch(si, (E, mb, mb), _e_gemm_nt_tri, (gX, gX, gA_t, mb, kb, NTM, NTN, NTK))
    sync()
    d1 = int((gA_f.to_numpy() != gA_t.to_numpy()).sum())
    print(f"[b] _e_gemm_nt_tri vs full (E={E}, {M}x{N} → {M}²): bit-diffs = {d1}  {'OK' if d1 == 0 else 'FAIL'}")

    # fused axpy on the symmetric A: B = b·A + c·A@A
    gB_f, gB_t = Z16(E * M, M), Z16(E * M, M)
    kbm = M // NTK
    b, c = -4.7750, 2.0315
    ct.launch(si, (E, mb, mb), _e_gemm_axpy, (gA_f, gA_f, gA_f, gB_f, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK))
    ct.launch(si, (E, mb, mb), _e_gemm_axpy_tri, (gA_f, gA_f, gA_f, gB_t, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK))
    sync()
    d2 = int((gB_f.to_numpy() != gB_t.to_numpy()).sum())
    print(f"[c] _e_gemm_axpy_tri vs full (symmetric A@A): bit-diffs = {d2}  {'OK' if d2 == 0 else 'FAIL'}")

    # timing: the two symmetric GEMMs, tri vs full
    def tim(kern, args, grid, reps=50):
        ct.launch(si, grid, kern, args); sync()
        t = time.perf_counter()
        for _ in range(reps): ct.launch(si, grid, kern, args)
        sync(); return (time.perf_counter() - t) / reps * 1e6
    t_nt_f = tim(_e_gemm_nt, (gX, gX, gA_f, mb, kb, NTM, NTN, NTK), (E, mb, mb))
    t_nt_t = tim(_e_gemm_nt_tri, (gX, gX, gA_t, mb, kb, NTM, NTN, NTK), (E, mb, mb))
    t_ax_f = tim(_e_gemm_axpy, (gA_f, gA_f, gA_f, gB_f, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK), (E, mb, mb))
    t_ax_t = tim(_e_gemm_axpy_tri, (gA_f, gA_f, gA_f, gB_t, float(b), float(c), mb, kbm, kbm, NTM, NTN, NTK), (E, mb, mb))
    print(f"[d] X@Xᵀ  : full {t_nt_f:6.0f} us → tri {t_nt_t:6.0f} us ({t_nt_f/t_nt_t:.2f}×)")
    print(f"    A@A ax: full {t_ax_f:6.0f} us → tri {t_ax_t:6.0f} us ({t_ax_f/t_ax_t:.2f}×)")
    for g in (gX, gA_f, gA_t, gB_f, gB_t): g.free()
    return d1 == 0 and d2 == 0


if __name__ == "__main__":
    print("=" * 84)
    print("Triangular (symmetric-output) NS GEMMs — runtime-if probe + bitwise + timing")
    print("=" * 84)
    ok = probe_if()
    if ok:
        ok &= probe_tri_gemms()
    print("=" * 84)
    print("  PASS" if ok else "  FAIL")
