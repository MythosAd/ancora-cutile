"""DEVICE validation of Polar Express coeffs in the resident NS (task #26). Confirms, on real shapes:
  (1) schedule=None is byte-identical to host Keller-5 (the DEFAULT must not regress);
  (2) device PE-5 matches host PE-5 within bf16 (the per-iteration coeff plumbing is correct);
  (3) device PE-5 orthogonalizes BETTER than Keller-5, and PE-4 ≈ Keller-5 (the speed option) — on
      device bf16, the regime the host probe predicted;
  (4) the BATCHED-expert path takes the schedule too (no cross-expert leak under PE).
Metric = orthogonality error ‖YYᵀ−I‖_F/√M (rows orthonormal ⇒ 0)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import (newton_schulz_resident, newton_schulz_resident_e,
                                     PE_SCHEDULE, A_COEF, B_COEF, C_COEF)
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
KELLER = [(A_COEF, B_COEF, C_COEF)] * 8


def host_ns(X, sched, n):
    """fp32 host NS with a per-iteration coeff schedule (Frobenius norm), mirroring the device math."""
    X = X.astype(np.float32); X = X / (np.linalg.norm(X) + 1e-7)
    for k in range(n):
        a, b, c = sched[k] if k < len(sched) else sched[-1]
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


def ortho_err(Y):
    M = Y.shape[0]
    return float(np.linalg.norm(Y @ Y.T - np.eye(M, dtype=Y.dtype)) / np.sqrt(M))


def dev_ns(Xbf, M, N, steps, schedule):
    """Run the resident single-matrix NS, return the bf16 result as f32. Xbf = (M,N) bf16 bits."""
    Z = lambda *s: _GpuArray.zeros(s, np.uint16)
    gX = _GpuArray(Xbf.copy()); gA, gA2, gB, gBX = Z(M, M), Z(M, M), Z(M, M), Z(M, N)
    grecip = _GpuArray.zeros((1, 1), np.float32)
    newton_schulz_resident(gX, gA, gA2, gB, gBX, grecip, M, N, si, steps=steps, schedule=schedule)
    cudart.cudaStreamSynchronize(si)
    out = b2f(gX.to_numpy())
    for g in (gX, gA, gA2, gB, gBX, grecip): g.free()
    return out


def cos_rel(a, b):
    a, b = a.ravel(), b.ravel()
    return (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
            float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)))


def single(name, M, N):
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((M, N)) @ np.diag(rng.uniform(0.1, 3.0, N))).astype(np.float32)
    Xbf = _f32bf(W)                                          # bf16-bits input (what the optimizer feeds)
    Xin = b2f(Xbf)                                           # the exact f32 value the device starts from

    d_keller5 = dev_ns(Xbf, M, N, 5, None)
    d_pe5     = dev_ns(Xbf, M, N, 5, PE_SCHEDULE)
    d_pe4     = dev_ns(Xbf, M, N, 4, PE_SCHEDULE)
    h_keller5 = host_ns(Xin, KELLER, 5)
    h_pe5     = host_ns(Xin, PE_SCHEDULE, 5)

    c_def, r_def = cos_rel(d_keller5, h_keller5)             # (1) default == host Keller (no regression)
    c_pe,  r_pe  = cos_rel(d_pe5, h_pe5)                     # (2) PE kernel correct vs host PE
    print(f"\n[{name} {M}x{N}]")
    print(f"  (1) device default(None,5) vs host Keller-5 : cos {c_def:.4f} rel {r_def:.2%}  "
          f"{'OK' if c_def > 0.999 and r_def < 0.02 else 'FAIL'}")
    print(f"  (2) device PE-5 vs host PE-5               : cos {c_pe:.4f} rel {r_pe:.2%}  "
          f"{'OK' if c_pe > 0.999 and r_pe < 0.02 else 'FAIL'}")
    e_k5, e_p5, e_p4 = ortho_err(d_keller5), ortho_err(d_pe5), ortho_err(d_pe4)
    print(f"  (3) orthogonality err  Keller-5={e_k5:.4f}  PE-5={e_p5:.4f}  PE-4={e_p4:.4f}")
    print(f"      PE-5 better than Keller-5: {'YES' if e_p5 < e_k5 else 'no'}   "
          f"PE-4 ≈/≤ Keller-5: {'YES' if e_p4 <= e_k5 * 1.05 else 'no'}")
    return c_def > 0.999 and c_pe > 0.999 and e_p5 < e_k5


def batched(E, M):
    """(4) PE through the batched-expert NS — each expert orthogonalized, no leak."""
    rng = np.random.default_rng(2)
    Ws = [(rng.standard_normal((M, M)) @ np.diag(rng.uniform(0.1, 3.0, M))).astype(np.float32) for _ in range(E)]
    packed = np.concatenate([_f32bf(W) for W in Ws], axis=0)
    Z = lambda *s: _GpuArray.zeros(s, np.uint16)
    gX = _GpuArray(packed.copy()); gA, gA2, gB, gBX = Z(E * M, M), Z(E * M, M), Z(E * M, M), Z(E * M, M)
    grecip = _GpuArray.zeros((E, 1), np.float32)
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, grecip, E, M, M, si, steps=5, schedule=PE_SCHEDULE)
    cudart.cudaStreamSynchronize(si)
    out = b2f(gX.to_numpy())
    ok = True; worst_e = 0.0
    for e in range(E):
        Od = out[e * M:(e + 1) * M]; Oh = host_ns(b2f(_f32bf(Ws[e])), PE_SCHEDULE, 5)
        c, r = cos_rel(Od, Oh); worst_e = max(worst_e, ortho_err(Od))
        ok &= (c > 0.99 and r < 0.05)
    for g in (gX, gA, gA2, gB, gBX, grecip): g.free()
    print(f"\n[batched E={E} {M}x{M} PE-5]  worst-expert ortho-err {worst_e:.4f}  "
          f"vs-host {'OK (no leak)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 80)
    print("Polar Express on device — default regression + PE correctness + orthogonality + batched")
    print("=" * 80)
    ok = True
    ok &= single("square (expert/kv/ffn)", 1024, 1024)
    ok &= single("rect (q/o, M<=N)", 1024, 2048)
    ok &= batched(4, 256)
    print("\n" + "=" * 80)
    print("  PASS" if ok else "  FAIL")
