"""HOST de-risk for Polar Express coefficients (task #25), BEFORE any device kernel.

Muon's NS orthogonalizes a momentum matrix toward its polar factor U@Vᵀ (all singular values → 1).
Our device NS uses fixed Keller coeffs (3.4445,-4.7750,2.0315) × 5 under FROBENIUS normalization
(X/‖X‖_F). Polar Express (Amsel et al. 2505.16932, schedule from Dao-AILab/gram-newton-schulz) uses
a DIFFERENT minimax-optimal (a,b,c) PER iteration. The catch: PE is derived assuming the SPECTRAL norm
is normalized to 1, but we normalize by Frobenius (σ_max ≤ 1, usually < 1). This probe answers, on the
host in fp32 (no GPU), the only questions that matter before writing kernels:
  (1) do the PE coeffs actually orthogonalize under OUR Frobenius normalization?
  (2) how few PE iters match/beat Keller-5 (the speed thesis = fewer iters)?
  (3) does a frobenius vs spectral-ish (×√M safety) normalization matter?
Metric = orthogonality error ‖Y Yᵀ − I‖_F/√M and the singular-value spread of Y (want all ≈ 1)."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

KELLER = [(3.4445, -4.7750, 2.0315)] * 8                       # fixed, repeated
# Polar Express 5-step schedule (Dao-AILab/gram-newton-schulz). Aggressive early → converged tail.
PE = [(4.0848, -6.8946, 2.9270),
      (3.9505, -6.3029, 2.6377),
      (3.7418, -5.5913, 2.3037),
      (2.8769, -3.1427, 1.2046),
      (2.8366, -3.0525, 1.2012)]
PE_TAIL = PE[-1]                                               # the converged coeff to repeat past step 5


def ns(X, coeffs, n, norm="fro"):
    """Run n NS steps with the given coeff schedule. X (M,N), M≤N. fp32 to mimic device math closely."""
    X = X.astype(np.float32)
    if norm == "fro":
        X = X / (np.linalg.norm(X) + 1e-7)
    elif norm == "spec":                                      # exact spectral (σ_max=1) — PE's assumption
        X = X / (np.linalg.svd(X, compute_uv=False)[0] + 1e-7)
    for k in range(n):
        a, b, c = coeffs[k] if k < len(coeffs) else PE_TAIL
        A = X @ X.T                                            # (M,M) gram
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


def ortho_err(Y):
    """‖YYᵀ − I‖_F/√M (rows orthonormal ⇒ 0) and the min/max singular value of Y (want [1,1])."""
    M = Y.shape[0]
    e = np.linalg.norm(Y @ Y.T - np.eye(M, dtype=Y.dtype)) / np.sqrt(M)
    s = np.linalg.svd(Y, compute_uv=False)
    return e, s.min(), s.max()


def make(kind, M, N, rng):
    """Test matrices spanning the conditioning a real momentum hits."""
    if kind == "gauss":
        return rng.standard_normal((M, N))
    if kind == "illcond":                                     # decaying spectrum (cond ~1e3)
        U, _ = np.linalg.qr(rng.standard_normal((M, M)))
        V, _ = np.linalg.qr(rng.standard_normal((N, N)))
        s = np.logspace(0, -3, M)
        return (U * s) @ V[:M]
    if kind == "lowrank":                                     # near-low-rank (hard for NS)
        return rng.standard_normal((M, 8)) @ rng.standard_normal((8, N))


def main():
    rng = np.random.default_rng(0)
    shapes = [("square", 256, 256), ("rect2x", 256, 512)]     # expert-like + q/o-like aspect ratios
    print("=" * 92)
    print("Polar Express vs Keller — orthogonality error ‖YYᵀ−I‖/√M (lower=better), σ-spread [min,max]")
    print("=" * 92)
    for sh, M, N in shapes:
        for kind in ("gauss", "illcond", "lowrank"):
            X = make(kind, M, N, rng)
            tgt_err, *_ = ortho_err(np.linalg.svd(X, full_matrices=False)[0] @ np.linalg.svd(X, full_matrices=False)[2])
            print(f"\n[{sh} {M}x{N}  {kind}]  (SVD polar-factor floor err={tgt_err:.1e})")
            for n in (3, 4, 5, 6):
                ek, kmin, kmax = ortho_err(ns(X, KELLER, n, "fro"))
                ep, pmin, pmax = ortho_err(ns(X, PE,     n, "fro"))
                print(f"   n={n}:  Keller err={ek:7.4f} σ[{kmin:.3f},{kmax:.3f}]   "
                      f"PE/fro err={ep:7.4f} σ[{pmin:.3f},{pmax:.3f}]")
            # does spectral normalization change PE's verdict at n=5?
            es, smin, smax = ortho_err(ns(X, PE, 5, "spec"))
            print(f"   n=5:  PE/spec err={es:7.4f} σ[{smin:.3f},{smax:.3f}]  (PE's native normalization)")
    print("\n" + "=" * 92)
    print("Read: if PE/fro reaches Keller-5's err at n=3 or n=4, the 'fewer iters' speed thesis holds")
    print("under our Frobenius normalization. If PE/fro is WORSE than PE/spec, normalization matters.")


if __name__ == "__main__":
    main()
