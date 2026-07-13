"""Probe the NS-chain fusion kernels + the RECTANGULAR batched NS (task #30/#32) BEFORE wiring them.

  (1) _gemm_axpy_bf16 / _e_gemm_axpy PER-TILE correctness vs a host f32 reference — the pitfall-0c
      lesson: a silent one-tile miscompile hides in an aggregate norm, so check every 128×128 tile.
      D==A aliasing (the B = b·A + c·A@A case) is exercised explicitly.
  (2) _muon_mom_t == _muon_mom + _transpose_mat and _muon_update_cast(_t) == _muon_update + _cast_bf16
      BIT-exact (same math, one launch) — these replace the per-weight staging for tall q/o.
  (3) RECT batched NS (E matrices (1024,2048) packed) vs the single-matrix NS per slice — the driver
      was square-only by docstring; validate the rectangular path end-to-end.
  (4) fused vs unfused NS: orthogonality quality (must be ≈ or better) + wall-time (the win).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import (
    newton_schulz_resident, newton_schulz_resident_e, _gemm_axpy_bf16, _e_gemm_axpy,
    _muon_mom, _muon_mom_t, _muon_update, _muon_update_cast, _muon_update_cast_t,
    _transpose_mat, NTM, NTN, NTK)
from ancora.kernels.fused import _cast_bf16, RTM, RTN
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
Z16 = lambda *s: _GpuArray.zeros(s, np.uint16)


def tile_err(got, ref, T=128):
    """Max per-TILE relative error — every (T,T) tile must be individually small (pitfall 0c)."""
    M, N = ref.shape; worst = 0.0
    for m in range(0, M, T):
        for n in range(0, N, T):
            r = ref[m:m+T, n:n+T]; g = got[m:m+T, n:n+T]
            d = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-30)
            worst = max(worst, d)
    return worst


def probe_gemm_axpy():
    """(1) fused GEMM+axpy vs host f32 reference, per tile. D==A alias case."""
    rng = np.random.default_rng(0); M = 512
    A = _f32bf(rng.standard_normal((M, M)).astype(np.float32) * 0.5)
    Af = b2f(A)
    gA, gC = _GpuArray(A.copy()), Z16(M, M)
    b, c = -4.7750, 2.0315
    # single-matrix: C = b·A + c·(A@A), D aliases A
    ct.launch(si, (M // NTM, M // NTN, 1), _gemm_axpy_bf16,
              (gA, gA, gA, gC, float(b), float(c), M // NTK, NTM, NTN, NTK))
    sync()
    ref = b * Af + c * (Af @ Af)                       # f32 host ref (fused keeps the product f32)
    e1 = tile_err(b2f(gC.to_numpy()), ref)
    # batched E=4, D==A alias
    E = 4
    Ab = np.concatenate([_f32bf(rng.standard_normal((M, M)).astype(np.float32) * 0.5) for _ in range(E)], 0)
    gAb, gCb = _GpuArray(Ab.copy()), Z16(E * M, M)
    mb = M // NTM
    ct.launch(si, (E, mb, M // NTN), _e_gemm_axpy,
              (gAb, gAb, gAb, gCb, float(b), float(c), mb, M // NTK, M // NTK, NTM, NTN, NTK))
    sync()
    out = b2f(gCb.to_numpy()); e2 = 0.0
    for e in range(E):
        Ae = b2f(Ab[e*M:(e+1)*M])
        e2 = max(e2, tile_err(out[e*M:(e+1)*M], b * Ae + c * (Ae @ Ae)))
    ok = e1 < 0.02 and e2 < 0.02
    print(f"[1] fused GEMM+axpy per-tile vs f32 host: single {e1:.4f}  batched(E={E}) {e2:.4f}  "
          f"{'OK' if ok else 'FAIL'}")
    for g in (gA, gC, gAb, gCb): g.free()
    return ok


def probe_mom_update():
    """(2) transposed momentum + fused update/cast must be BIT-identical to the unfused pairs."""
    rng = np.random.default_rng(1); K, N = 2048, 1024   # the o_proj (tall) shape
    buf0 = rng.standard_normal((K, N)).astype(np.float32)
    g0 = rng.standard_normal((K, N)).astype(np.float32)
    # reference: _muon_mom → u (K,N) → _transpose_mat → uT (N,K)
    bufA, gG = _GpuArray(buf0.copy()), _GpuArray(g0.copy())
    uKN, uT_ref = Z16(K, N), Z16(N, K)
    ct.launch(si, (K // NTM, N // NTN, 1), _muon_mom, (bufA, gG, uKN, 0.95, NTM, NTN))
    ct.launch(si, (K // 64, N // 64, 1), _transpose_mat, (uKN, uT_ref, 64))
    # fused: _muon_mom_t → uT directly
    bufB, uT = _GpuArray(buf0.copy()), Z16(N, K)
    ct.launch(si, (K // 64, N // 64, 1), _muon_mom_t, (bufB, gG, uT, 0.95, 64))
    sync()
    d_buf = np.abs(bufA.to_numpy() - bufB.to_numpy()).max()
    d_u = int((uT_ref.to_numpy() != uT.to_numpy()).sum())
    ok1 = d_buf == 0.0 and d_u == 0
    print(f"[2a] _muon_mom_t vs mom+transpose: buf max|Δ|={d_buf:.0e}  uT bit-diffs={d_u}  {'OK' if ok1 else 'FAIL'}")

    # update+cast: reference _muon_update + _cast_bf16 vs fused _muon_update_cast (non-transposed)
    p0 = rng.standard_normal((K, N)).astype(np.float32)
    O = _f32bf(rng.standard_normal((K, N)).astype(np.float32))
    pA, pB = _GpuArray(p0.copy()), _GpuArray(p0.copy())
    p16A, p16B, gO = Z16(K, N), Z16(K, N), _GpuArray(O.copy())
    ct.launch(si, (K // NTM, N // NTN, 1), _muon_update, (pA, gO, 0.02, NTM, NTN))
    ct.launch(si, (K // RTM, N // RTN, 1), _cast_bf16, (pA, p16A))
    ct.launch(si, (K // NTM, N // NTN, 1), _muon_update_cast, (pB, p16B, gO, 0.02, NTM, NTN))
    sync()
    d_p = np.abs(pA.to_numpy() - pB.to_numpy()).max()
    d_16 = int((p16A.to_numpy() != p16B.to_numpy()).sum())
    ok2 = d_p == 0.0 and d_16 == 0
    print(f"[2b] _muon_update_cast vs update+cast: p32 max|Δ|={d_p:.0e}  p16 bit-diffs={d_16}  {'OK' if ok2 else 'FAIL'}")

    # transposed variant: O lives as (N,K); update p32 (K,N)
    OT = np.ascontiguousarray(O.T)                       # bit-transpose of the bf16 bits
    pC, p16C, gOT = _GpuArray(p0.copy()), Z16(K, N), _GpuArray(OT.copy())
    ct.launch(si, (K // 64, N // 64, 1), _muon_update_cast_t, (pC, p16C, gOT, 0.02, 64))
    sync()
    d_pt = np.abs(pA.to_numpy() - pC.to_numpy()).max()
    d_16t = int((p16A.to_numpy() != p16C.to_numpy()).sum())
    ok3 = d_pt == 0.0 and d_16t == 0
    print(f"[2c] _muon_update_cast_t (transposed O): p32 max|Δ|={d_pt:.0e}  p16 bit-diffs={d_16t}  {'OK' if ok3 else 'FAIL'}")
    for g in (bufA, bufB, gG, uKN, uT_ref, uT, pA, pB, pC, p16A, p16B, p16C, gO, gOT): g.free()
    return ok1 and ok2 and ok3


def run_single(Xbf, M, N, steps, fused):
    gX = _GpuArray(Xbf.copy())
    gA, gA2, gB, gBX = Z16(M, M), Z16(M, M), Z16(M, M), Z16(M, N)
    rec = _GpuArray.zeros((1, 1), np.float32)
    newton_schulz_resident(gX, gA, gA2, gB, gBX, rec, M, N, si, steps=steps, schedule=None, fused=fused)
    sync(); out = b2f(gX.to_numpy())
    for g in (gX, gA, gA2, gB, gBX, rec): g.free()
    return out


def probe_rect_batched(E=3, M=1024, N=2048, steps=5):
    """(3) rect batched NS slice-vs-single + (4a) fused-vs-unfused quality."""
    rng = np.random.default_rng(2)
    Ws = [_f32bf((rng.standard_normal((M, N)) @ np.diag(rng.uniform(0.1, 3.0, N))).astype(np.float32))
          for _ in range(E)]
    packed = np.concatenate(Ws, 0)
    gX = _GpuArray(packed.copy())
    gA, gA2, gB, gBX = Z16(E * M, M), Z16(E * M, M), Z16(E * M, M), Z16(E * M, N)
    rec = _GpuArray.zeros((E, 1), np.float32)
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, rec, E, M, N, si, steps=steps, schedule=None, fused=True)
    sync(); out = b2f(gX.to_numpy())
    ok = True; worst_r, worst_o, worst_o_unf = 0.0, 0.0, 0.0
    for e in range(E):
        got = out[e*M:(e+1)*M]
        ref_f = run_single(Ws[e], M, N, steps, fused=True)      # single-matrix fused
        ref_u = run_single(Ws[e], M, N, steps, fused=False)     # single-matrix unfused (the old chain)
        r = np.linalg.norm(got - ref_f) / (np.linalg.norm(ref_f) + 1e-30)
        worst_r = max(worst_r, float(r))
        oe = np.linalg.norm(got @ got.T - np.eye(M, dtype=np.float32)) / np.sqrt(M)
        ou = np.linalg.norm(ref_u @ ref_u.T - np.eye(M, dtype=np.float32)) / np.sqrt(M)
        worst_o, worst_o_unf = max(worst_o, float(oe)), max(worst_o_unf, float(ou))
        ok &= r < 0.01
    print(f"[3] RECT batched (E={E},{M}x{N}) vs single-matrix NS: worst rel {worst_r:.2%}  {'OK' if ok else 'FAIL'}")
    print(f"[4a] ortho-err fused {worst_o:.4f} vs unfused {worst_o_unf:.4f}  "
          f"{'OK (≈/better)' if worst_o <= worst_o_unf * 1.10 else 'WORSE'}")
    for g in (gX, gA, gA2, gB, gBX, rec): g.free()
    return ok and worst_o <= worst_o_unf * 1.10


def probe_square_quality(M=1024, steps=5):
    """(4a') square fused-vs-unfused quality on the expert shape."""
    rng = np.random.default_rng(3)
    W = _f32bf((rng.standard_normal((M, M)) @ np.diag(rng.uniform(0.1, 3.0, M))).astype(np.float32))
    yf = run_single(W, M, M, steps, fused=True)
    yu = run_single(W, M, M, steps, fused=False)
    of = np.linalg.norm(yf @ yf.T - np.eye(M, dtype=np.float32)) / np.sqrt(M)
    ou = np.linalg.norm(yu @ yu.T - np.eye(M, dtype=np.float32)) / np.sqrt(M)
    rel = np.linalg.norm(yf - yu) / (np.linalg.norm(yu) + 1e-30)
    ok = of <= ou * 1.10 and rel < 0.05
    print(f"[4a'] square {M}²: ortho-err fused {of:.4f} vs unfused {ou:.4f}  fused-vs-unfused rel {rel:.2%}  "
          f"{'OK' if ok else 'FAIL'}")
    return bool(ok)


def time_chain(E, M, N, fused, steps=5, reps=20):
    rng = np.random.default_rng(0)
    packed = np.concatenate([_f32bf(rng.standard_normal((M, N)).astype(np.float32)) for _ in range(E)], 0)
    gX = _GpuArray(packed.copy())
    gA, gA2, gB, gBX = Z16(E * M, M), Z16(E * M, M), Z16(E * M, M), Z16(E * M, N)
    rec = _GpuArray.zeros((E, 1), np.float32)
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, rec, E, M, N, si, steps=steps, schedule=None, fused=fused)
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        newton_schulz_resident_e(gX, gA, gA2, gB, gBX, rec, E, M, N, si, steps=steps, schedule=None, fused=fused)
    sync(); us = (time.perf_counter() - t) / reps * 1e6
    for g in (gX, gA, gA2, gB, gBX, rec): g.free()
    return us


if __name__ == "__main__":
    print("=" * 84)
    print("NS fusion + rectangular batched NS — device probe")
    print("=" * 84)
    ok = True
    ok &= probe_gemm_axpy()
    ok &= probe_mom_update()
    ok &= probe_rect_batched()
    ok &= probe_square_quality()
    t_u16 = time_chain(16, 1024, 1024, fused=False)
    t_f16 = time_chain(16, 1024, 1024, fused=True)
    t_u24 = time_chain(24, 1024, 2048, fused=False)
    t_f24 = time_chain(24, 1024, 2048, fused=True)
    print(f"[4b] NS chain wall-time:")
    print(f"       square E=16 1024²    : unfused {t_u16:6.0f} us → fused {t_f16:6.0f} us ({t_u16/t_f16:.2f}×)")
    print(f"       rect   E=24 1024x2048: unfused {t_u24:6.0f} us → fused {t_f24:6.0f} us ({t_u24/t_f24:.2f}×)")
    print("=" * 84)
    print("  PASS" if ok else "  FAIL")
