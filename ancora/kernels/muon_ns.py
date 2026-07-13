"""
ancora/kernels/muon_ns.py — RESIDENT Newton-Schulz for the device Muon optimizer.

The quintic NS iteration (Keller Jordan coeffs a,b,c) orthogonalizes a momentum matrix:
  X = X / ‖X‖_F
  repeat 5×:  A = X@Xᵀ ;  B = b·A + c·A² ;  X = a·X + B@X
Done in BF16 ON GPU (Recursive/nanochat/Kimi all do bf16 NS) and — unlike the round-trip
`muon.newton_schulz5_device` (2.5× SLOWER: alloc/upload/download/sync PER matmul) — CHAINED on
PERSISTENT device buffers: upload the momentum once, run the 15 GEMMs + axpy/norm on-device,
read the result once. This is the missing piece muon.py flagged ("a real win needs a RESIDENT NS").

Matrices are (M,N) with M ≤ N (the caller transposes a tall weight first). All dims must be
128/64 tile-aligned (the proj/expert weight shapes are). Output overwrites gX in place."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import cuda.tile as ct
import ancora.env  # noqa: F401
from ancora.kernels.fused import _gemm_bf16

NTM, NTN, NTK = 128, 128, 64
A_COEF, B_COEF, C_COEF = 3.4445, -4.7750, 2.0315

# Polar Express minimax-optimal per-iteration coeffs (Amsel et al. 2505.16932; schedule from
# Dao-AILab/gram-newton-schulz). Each entry is one NS iteration's (a,b,c) for X = a·X + (b·A + c·A²)@X.
# Drop-in for Keller under OUR Frobenius normalization — host-validated (tests/kernels/_probe_polar_express.py):
# at 5 iters PE beats Keller everywhere (e.g. rect-gauss orthogonality err 0.31→0.034); PE-4 ≈ Keller-5 on
# well-conditioned inputs (the ~20% fewer-matmul speed option), marginally worse on ill-conditioned. Past
# the listed steps the last (converged) triple repeats. Pass schedule=PE_SCHEDULE to opt in; default None
# keeps Keller for ALL steps → byte-identical to the original NS (regression-safe).
PE_SCHEDULE = [(4.0848, -6.8946, 2.9270),
               (3.9505, -6.3029, 2.6377),
               (3.7418, -5.5913, 2.3037),
               (2.8769, -3.1427, 1.2046),
               (2.8366, -3.0525, 1.2012)]


def _coef_at(schedule, k):
    """The (a,b,c) for NS iteration k: Keller (fixed) if schedule is None, else schedule[k] with the
    last triple repeated once the schedule is exhausted (PE's converged coeff)."""
    if schedule is None:
        return A_COEF, B_COEF, C_COEF
    return schedule[k] if k < len(schedule) else schedule[-1]


# Process-wide NS coefficient default. The production Muon classes (DeviceMuon/ResidentMuon/
# BatchedProjMuon/GroupedMoEFFN) call the drivers WITHOUT an explicit `schedule`, so this one toggle
# selects Keller (None) vs Polar Express for ALL of them with zero per-constructor threading — fine
# for a single-model-per-process framework. Default None ⇒ Keller everywhere ⇒ byte-identical to the
# original NS (regression-safe). `schedule=None` passed EXPLICITLY still means Keller (the _KELLER
# sentinel distinguishes "caller said Keller" from "caller said nothing, use the default").
_KELLER = object()
_DEFAULT_SCHEDULE = None


def set_polar_express(on: bool):
    """Globally select Polar Express (on) vs Keller (off) for every Muon NS call that doesn't pass an
    explicit schedule. Call once at startup. Returns the previous setting (for save/restore in tests)."""
    global _DEFAULT_SCHEDULE
    prev = _DEFAULT_SCHEDULE
    _DEFAULT_SCHEDULE = PE_SCHEDULE if on else None
    return prev


def _resolve(schedule):
    return _DEFAULT_SCHEDULE if schedule is _KELLER else schedule


@ct.kernel(occupancy=2)
def _gemm_nt_bf16(A, B, C, KB: ct.Constant[int],
                  TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = A @ Bᵀ, BF16-bit inputs, f32 accumulate, BF16-bit OUTPUT. A(M,K), B(N,K) → C(M,N).
    (= fused._gemm_nt_f32 but bf16 out — for X@Xᵀ in the NS chain.)"""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, ct.transpose(tb), acc)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _axpy_bf16(out, A, B, alpha, beta, TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """out = alpha·A + beta·B (BF16 bits, f32 math). alpha/beta runtime floats. Grid (M//TM,N//TN)."""
    m, n = ct.bid(0), ct.bid(1)
    a = ct.astype(ct.bitcast(ct.load(A, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    b = ct.astype(ct.bitcast(ct.load(B, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(alpha * a + beta * b, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _gemm_axpy_bf16(A, B, D, C, alpha, beta, KB: ct.Constant[int],
                    TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = alpha·D + beta·(A @ B) — the NS axpy FUSED into the GEMM epilogue (bf16 in/out, f32 acc;
    the A@B product stays f32 in the accumulator — one fewer bf16 rounding than the unfused pair).
    C must NOT alias A or B (other blocks still read them); D is only read at this block's own (m,n)
    tile so D==A is safe. Grid (M//TM, N//TN)."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    d = ct.astype(ct.bitcast(ct.load(D, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(alpha * d + beta * acc, ct.bfloat16), ct.uint16))


# ── TRIANGULAR (symmetric-output) variants: A = X@Xᵀ and B = b·A + c·A@A (A symmetric) are both
# SYMMETRIC, so only the LOWER-triangle blocks (n ≤ m) compute; a separate tiny _mirror pass copies
# the strict-lower transposes to the upper half. The mirror is BIT-identical to computing the upper
# blocks directly (same products, same k order; IEEE multiply commutes bitwise; probed
# _probe_ns_tri3.py: 0 bit-diffs, mirror ≈ 17 µs vs 450 µs for an in-GEMM transpose(acc) store —
# the mma-fragment layout scatters, a load→transpose→store copy coalesces). ~44% of those two
# GEMMs' blocks skipped → 1.5-1.7× each (the runtime `if bid` guard is a REAL branch, not
# predication: idle blocks retire immediately). Requires TM==TN. ──

@ct.kernel(occupancy=2)
def _gemm_nt_tril(A, B, C, KB: ct.Constant[int],
                  TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = A@Bᵀ, LOWER-triangle blocks only (n≤m; symmetric output — run _mirror after)."""
    m, n = ct.bid(0), ct.bid(1)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, ct.transpose(tb), acc)
        ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _gemm_axpy_tril(A, B, D, C, alpha, beta, KB: ct.Constant[int],
                    TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """_gemm_axpy_bf16, LOWER-triangle blocks only (A,B,D symmetric ⇒ C symmetric; _mirror after)."""
    m, n = ct.bid(0), ct.bid(1)
    if n <= m:
        acc = ct.zeros((TM_, TN_), ct.float32)
        for k in range(KB):
            ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
            tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
            acc = ct.mma(ta, tb, acc)
        d = ct.astype(ct.bitcast(ct.load(D, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
        ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(alpha * d + beta * acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _mirror(C, TT_: ct.Constant[int]):
    """C[m,n] ← C[n,m]ᵀ for n>m (bit copy; completes a lower-triangle symmetric result).
    Grid (M//TT, M//TT)."""
    m, n = ct.bid(0), ct.bid(1)
    if n > m:
        t = ct.load(C, index=(n, m), shape=(TT_, TT_))
        ct.store(C, index=(m, n), tile=ct.transpose(t))


@ct.kernel
def _fro_recip(X, recip, MB: ct.Constant[int], NB: ct.Constant[int],
               TM_: ct.Constant[int], TN_: ct.Constant[int], eps: ct.Constant[float]):
    """recip = 1/(‖X‖_F + eps), one block summing X² over all tiles. X bf16 bits; recip (1,1) f32."""
    acc = ct.zeros((1, 1), ct.float32)
    for m in range(MB):
        for n in range(NB):
            x = ct.astype(ct.bitcast(ct.load(X, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
            acc = acc + ct.sum(ct.sum(x * x, axis=-1, keepdims=True), axis=0, keepdims=True)
    ct.store(recip, index=(0, 0), tile=1.0 / (ct.sqrt(acc) + eps))


@ct.kernel
def _scale(X, recip, TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """X *= recip (broadcast the (1,1) scalar). In place. Grid (M//TM, N//TN)."""
    m, n = ct.bid(0), ct.bid(1)
    r = ct.reshape(ct.load(recip, index=(0, 0), shape=(1, 1)), ())
    x = ct.astype(ct.bitcast(ct.load(X, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(X, index=(m, n), tile=ct.bitcast(ct.astype(x * r, ct.bfloat16), ct.uint16))


@ct.kernel
def _muon_mom(buf, g, u, mom, TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """Heavy-ball momentum + Nesterov look-ahead. buf,g f32; u BF16 bits (the NS input).
    buf ← mom·buf + g ;  u ← g + mom·buf_new. Grid (M//TM, N//TN)."""
    m, n = ct.bid(0), ct.bid(1)
    b = ct.load(buf, index=(m, n), shape=(TM_, TN_))
    gg = ct.load(g, index=(m, n), shape=(TM_, TN_))
    nb = mom * b + gg
    ct.store(buf, index=(m, n), tile=nb)
    ct.store(u, index=(m, n), tile=ct.bitcast(ct.astype(gg + mom * nb, ct.bfloat16), ct.uint16))


@ct.kernel
def _muon_update(p32, O, lr_scale, TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """fp32 master update: p32 ← p32 − lr_scale·O. O BF16 bits (the orthogonalized NS output)."""
    m, n = ct.bid(0), ct.bid(1)
    p = ct.load(p32, index=(m, n), shape=(TM_, TN_))
    o = ct.astype(ct.bitcast(ct.load(O, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(p32, index=(m, n), tile=p - lr_scale * o)


@ct.kernel
def _transpose_mat(X, XT, TT_: ct.Constant[int]):
    """XT = Xᵀ, BF16 bits. X(M,N) → XT(N,M). Grid (M//TT, N//TT)."""
    m, n = ct.bid(0), ct.bid(1)
    t = ct.bitcast(ct.load(X, index=(m, n), shape=(TT_, TT_)), ct.bfloat16)
    ct.store(XT, index=(n, m), tile=ct.bitcast(ct.transpose(t), ct.uint16))


@ct.kernel
def _muon_mom_t(buf, g, uT, mom, TT_: ct.Constant[int]):
    """_muon_mom with a TRANSPOSED u store — for TALL (K>N) weights whose NS input is uᵀ. buf,g f32
    (K,N); uT bf16 bits (N,K) (a packed-group slice). Same math/RNE as _muon_mom then transpose-store
    ⇒ bit-identical to _muon_mom + _transpose_mat, one launch + no staging. Grid (K//TT, N//TT)."""
    m, n = ct.bid(0), ct.bid(1)
    b = ct.load(buf, index=(m, n), shape=(TT_, TT_))
    gg = ct.load(g, index=(m, n), shape=(TT_, TT_))
    nb = mom * b + gg
    ct.store(buf, index=(m, n), tile=nb)
    u = ct.bitcast(ct.astype(gg + mom * nb, ct.bfloat16), ct.uint16)
    ct.store(uT, index=(n, m), tile=ct.transpose(u))


@ct.kernel
def _muon_update_cast(p32, p16, O, lr_scale, TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """fp32 master update + bf16 weight refresh in ONE pass: p32 ← p32 − lr_scale·O; p16 ← RNE(p32).
    (= _muon_update + fused._cast_bf16 without re-reading p32; same RNE ⇒ same p16 bits.)"""
    m, n = ct.bid(0), ct.bid(1)
    p = ct.load(p32, index=(m, n), shape=(TM_, TN_))
    o = ct.astype(ct.bitcast(ct.load(O, index=(m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    pn = p - lr_scale * o
    ct.store(p32, index=(m, n), tile=pn)
    ct.store(p16, index=(m, n), tile=ct.bitcast(ct.astype(pn, ct.bfloat16), ct.uint16))


@ct.kernel
def _muon_update_cast_t(p32, p16, O, lr_scale, TT_: ct.Constant[int]):
    """_muon_update_cast reading O TRANSPOSED — for tall weights whose NS output is Oᵀ (N,K) in the
    packed group. p32,p16 (K,N); O (N,K) slice. Grid (K//TT, N//TT)."""
    m, n = ct.bid(0), ct.bid(1)
    ot = ct.bitcast(ct.load(O, index=(n, m), shape=(TT_, TT_)), ct.bfloat16)
    o = ct.astype(ct.transpose(ot), ct.float32)
    p = ct.load(p32, index=(m, n), shape=(TT_, TT_))
    pn = p - lr_scale * o
    ct.store(p32, index=(m, n), tile=pn)
    ct.store(p16, index=(m, n), tile=ct.bitcast(ct.astype(pn, ct.bfloat16), ct.uint16))


def newton_schulz_resident(gX, gA, gA2, gB, gBX, grecip, M, N, si, steps=5, eps=1e-7, schedule=_KELLER,
                           fused=True):
    """Resident NS on device. gX (M,N) bf16 bits (the momentum, OVERWRITTEN with the orthogonalized
    result); gA/gA2/gB (M,M) bf16 scratch; gBX (M,N) bf16 scratch; grecip (1,1) f32. M ≤ N, both
    128-aligned, N divisible by 64. No host round-trip — the whole chain runs on `si`.
    schedule defaults to the process-wide setting (Keller unless set_polar_express(True)); pass
    schedule=None to force Keller, or schedule=PE_SCHEDULE for Polar Express per-iteration coeffs.
    fused=True (default): the two axpys ride the GEMM epilogues (5→3 launches/iter, no A²/BX HBM
    round-trip; X ping-pongs gX↔gBX with a copy home when steps is odd). Slightly DIFFERENT bits than
    unfused (A@A stays f32 in the fused combine — strictly fewer roundings); optimizer-only, forward
    untouched. fused=False reproduces the original 5-launch chain exactly."""
    schedule = _resolve(schedule)
    TM, TN, TK = NTM, NTN, NTK
    mb, nb = M // TM, N // TN
    ct.launch(si, (1, 1, 1), _fro_recip, (gX, grecip, mb, nb, TM, TN, float(eps)))
    ct.launch(si, (mb, nb, 1), _scale, (gX, grecip, TM, TN))
    if fused:
        X, Y = gX, gBX
        mm = M // 64                                           # mirror tile grid
        for k in range(steps):
            a, b, c = _coef_at(schedule, k)
            ct.launch(si, (mb, M // TN, 1), _gemm_nt_tril, (X, X, gA, N // TK, TM, TN, TK))  # A = X@Xᵀ (lower)
            ct.launch(si, (mm, mm, 1), _mirror, (gA, 64))                                     # A symmetric
            ct.launch(si, (mb, M // TN, 1), _gemm_axpy_tril,                                  # B = b·A + c·A@A (lower)
                      (gA, gA, gA, gB, float(b), float(c), M // TK, TM, TN, TK))
            ct.launch(si, (mm, mm, 1), _mirror, (gB, 64))                                     # B symmetric
            ct.launch(si, (mb, nb, 1), _gemm_axpy_bf16,                                       # Y = a·X + B@X (full)
                      (gB, X, X, Y, float(a), 1.0, M // TK, TM, TN, TK))
            X, Y = Y, X
        if X is not gX:                                        # odd steps → copy the result home
            ct.launch(si, (mb, nb, 1), _axpy_bf16, (gX, X, X, 1.0, 0.0, TM, TN))
        return
    for k in range(steps):
        a, b, c = _coef_at(schedule, k)
        ct.launch(si, (mb, M // TN, 1), _gemm_nt_bf16, (gX, gX, gA, N // TK, TM, TN, TK))    # A = X@Xᵀ  (M,M)
        ct.launch(si, (mb, M // TN, 1), _gemm_bf16,    (gA, gA, gA2, M // TK, TM, TN, TK))   # A² = A@A  (M,M)
        ct.launch(si, (mb, M // TN, 1), _axpy_bf16,    (gB, gA, gA2, float(b), float(c), TM, TN))  # B = b·A+c·A²
        ct.launch(si, (mb, nb, 1),      _gemm_bf16,    (gB, gX, gBX, M // TK, TM, TN, TK))   # BX = B@X  (M,N)
        ct.launch(si, (mb, nb, 1),      _axpy_bf16,    (gX, gX, gBX, float(a), 1.0, TM, TN)) # X = a·X + BX


# ── BATCHED-EXPERT Newton-Schulz (MoE expert weights: E SQUARE (M,M) matrices packed (E*M, M)) ──
# Muon on the MoE experts (Kimi/Moonshot recipe). The expert dim is bid(0); expert e is the row-block
# [e*(M//T), +M//T) (the _transpose_e/_ggemm pattern). ONE launch per stage covers ALL E experts → the
# same launch count as the single-matrix NS, not E×. Experts are uniform-SQUARE (Ie==H) so M==N==K and
# no per-expert transpose is needed. All these kernels require TM==TN (the per-expert row stride e*MB is
# shared by both A and B operands) — the NS always uses NTM==NTN==128, so that holds.

@ct.kernel(occupancy=2)
def _e_gemm_nt(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
               TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = A_e @ B_eᵀ (bf16 in/out, f32 acc). A,B packed (E*M, K); C packed (E*M, N).
    MB=M//TM (per-expert row-blocks). grid (E, M//TM, N//TN)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(e * MB + n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, ct.transpose(tb), acc)
    ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _e_gemm(A, B, C, MB: ct.Constant[int], KBe: ct.Constant[int], KB: ct.Constant[int],
            TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = A_e @ B_e (plain). A_e (M,K) packed (E*M,K) → MB=M//TM; B_e (K,N) packed (E*K,N) →
    KBe=K//TK (B's per-expert ROW-blocks). grid (E, M//TM, N//TN), K-loop KB=K//TK."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(e * KBe + k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _e_axpy(out, A, B, alpha, beta, MB: ct.Constant[int], TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """Batched out_e = alpha·A_e + beta·B_e (bf16 bits, f32 math). All packed (E*M, N), MB=M//TM.
    grid (E, M//TM, N//TN). alpha/beta runtime floats."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    a = ct.astype(ct.bitcast(ct.load(A, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    b = ct.astype(ct.bitcast(ct.load(B, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(out, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(alpha * a + beta * b, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _e_gemm_axpy(A, B, D, C, alpha, beta, MB: ct.Constant[int], KBe: ct.Constant[int], KB: ct.Constant[int],
                 TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched C_e = alpha·D_e + beta·(A_e @ B_e) — _e_gemm with the NS axpy fused into the epilogue
    (the product stays f32 in the accumulator). A packed (E*M,K), MB=M//TM; B packed (E*K,N),
    KBe=K//TK; D,C packed (E*M,N). C must NOT alias A/B; D==A is safe (own-tile read only).
    grid (E, M//TM, N//TN)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(e * MB + m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(e * KBe + k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    d = ct.astype(ct.bitcast(ct.load(D, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(C, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(alpha * d + beta * acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _e_gemm_nt_tril(A, B, C, MB: ct.Constant[int], KB: ct.Constant[int],
                    TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """Batched _gemm_nt_tril: C_e = A_e@B_eᵀ, lower-triangle blocks only (run _e_mirror after)."""
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
    """Batched _gemm_axpy_tril: C_e = alpha·D_e + beta·(A_e@B_e), lower blocks only (symmetric)."""
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
    """Batched _mirror: C_e[m,n] ← C_e[n,m]ᵀ for n>m. grid (E, M//TT, M//TT); MB=M//TT."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    if n > m:
        t = ct.load(C, index=(e * MB + n, m), shape=(TT_, TT_))
        ct.store(C, index=(e * MB + m, n), tile=ct.transpose(t))


@ct.kernel
def _e_fro_recip(X, recip, MB: ct.Constant[int], NB: ct.Constant[int],
                 TM_: ct.Constant[int], TN_: ct.Constant[int], eps: ct.Constant[float]):
    """Per-expert recip_e = 1/(‖X_e‖_F + eps). X packed (E*M,N); recip (E,1) f32. grid (E,1,1)."""
    e = ct.bid(0)
    acc = ct.zeros((1, 1), ct.float32)
    for m in range(MB):
        for n in range(NB):
            x = ct.astype(ct.bitcast(ct.load(X, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
            acc = acc + ct.sum(ct.sum(x * x, axis=-1, keepdims=True), axis=0, keepdims=True)
    ct.store(recip, index=(e, 0), tile=1.0 / (ct.sqrt(acc) + eps))


@ct.kernel
def _e_scale(X, recip, MB: ct.Constant[int], TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """X_e *= recip_e (broadcast the per-expert scalar). X packed (E*M,N); recip (E,1). grid (E, M//TM, N//TN)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    r = ct.reshape(ct.load(recip, index=(e, 0), shape=(1, 1)), ())
    x = ct.astype(ct.bitcast(ct.load(X, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(X, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(x * r, ct.bfloat16), ct.uint16))


@ct.kernel
def _e_muon_mom(buf, g, u, mom, MB: ct.Constant[int], TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """Batched heavy-ball+Nesterov momentum. buf,g f32 packed (E*M,N); u bf16 bits (the NS input).
    buf ← mom·buf + g ; u ← g + mom·buf_new. grid (E, M//TM, N//TN)."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    b = ct.load(buf, index=(e * MB + m, n), shape=(TM_, TN_))
    gg = ct.load(g, index=(e * MB + m, n), shape=(TM_, TN_))
    nb = mom * b + gg
    ct.store(buf, index=(e * MB + m, n), tile=nb)
    ct.store(u, index=(e * MB + m, n), tile=ct.bitcast(ct.astype(gg + mom * nb, ct.bfloat16), ct.uint16))


@ct.kernel
def _e_muon_update(p32, O, lr_scale, MB: ct.Constant[int], TM_: ct.Constant[int], TN_: ct.Constant[int]):
    """Batched master update p32_e ← p32_e − lr_scale·O_e. p32 f32 packed (E*M,N); O bf16 bits."""
    e, m, n = ct.bid(0), ct.bid(1), ct.bid(2)
    p = ct.load(p32, index=(e * MB + m, n), shape=(TM_, TN_))
    o = ct.astype(ct.bitcast(ct.load(O, index=(e * MB + m, n), shape=(TM_, TN_)), ct.bfloat16), ct.float32)
    ct.store(p32, index=(e * MB + m, n), tile=p - lr_scale * o)


def newton_schulz_resident_e(gX, gA, gA2, gB, gBX, grecip, E, M, N, si, steps=5, eps=1e-7, schedule=_KELLER,
                             fused=True):
    """Batched resident NS over E matrices (M,N) with M ≤ N, packed (E*M, N). SQUARE (experts,
    k/v/gate/up/down) and RECTANGULAR (q/o at (1024,2048), tall ones pre-transposed by the caller)
    both work — the (M,M) products and the (M,N) carry are sized separately. gX (E*M,N) bf16 bits
    (the momentum-Nesterov, OVERWRITTEN with the orthogonalized result); gA/gA2/gB (E*M,M) bf16
    scratch; gBX (E*M,N) bf16 scratch; grecip (E,1) f32. One chain, all matrices, no host round-trip.
    schedule defaults to the process-wide setting (Keller unless set_polar_express(True)); pass
    schedule=None to force Keller, or schedule=PE_SCHEDULE for Polar Express.
    fused=True: axpys fused into the GEMM epilogues (5→3 launches/iter, no A²/BX HBM round-trip;
    ping-pong gX↔gBX + copy home when steps is odd). fused=False = the original exact chain."""
    schedule = _resolve(schedule)
    TM, TN, TK = NTM, NTN, NTK
    mb, nb, kbm = M // TM, N // TN, M // TK
    ct.launch(si, (E, 1, 1), _e_fro_recip, (gX, grecip, mb, nb, TM, TN, float(eps)))
    ct.launch(si, (E, mb, nb), _e_scale, (gX, grecip, mb, TM, TN))
    if fused:
        X, Y = gX, gBX
        mm = M // 64                                          # mirror tile grid
        for k in range(steps):
            a, b, c = _coef_at(schedule, k)
            ct.launch(si, (E, mb, M // TN), _e_gemm_nt_tril, (X, X, gA, mb, N // TK, TM, TN, TK))  # A (lower)
            ct.launch(si, (E, mm, mm), _e_mirror, (gA, mm, 64))                                     # A symmetric
            ct.launch(si, (E, mb, M // TN), _e_gemm_axpy_tril,                                      # B (lower)
                      (gA, gA, gA, gB, float(b), float(c), mb, kbm, kbm, TM, TN, TK))
            ct.launch(si, (E, mm, mm), _e_mirror, (gB, mm, 64))                                     # B symmetric
            ct.launch(si, (E, mb, nb), _e_gemm_axpy,                                                # Y = a·X + B@X
                      (gB, X, X, Y, float(a), 1.0, mb, kbm, kbm, TM, TN, TK))
            X, Y = Y, X
        if X is not gX:                                       # odd steps → copy the result home
            ct.launch(si, (E, mb, nb), _e_axpy, (gX, X, X, 1.0, 0.0, mb, TM, TN))
        return
    for k in range(steps):
        a, b, c = _coef_at(schedule, k)
        ct.launch(si, (E, mb, M // TN), _e_gemm_nt, (gX, gX, gA, mb, N // TK, TM, TN, TK))      # A = X@Xᵀ (M,M)
        ct.launch(si, (E, mb, M // TN), _e_gemm,    (gA, gA, gA2, mb, kbm, kbm, TM, TN, TK))    # A² = A@A (M,M)
        ct.launch(si, (E, mb, M // TN), _e_axpy,    (gB, gA, gA2, float(b), float(c), mb, TM, TN))
        ct.launch(si, (E, mb, nb),      _e_gemm,    (gB, gX, gBX, mb, kbm, kbm, TM, TN, TK))    # BX = B@X (M,N)
        ct.launch(si, (E, mb, nb),      _e_axpy,    (gX, gX, gBX, float(a), 1.0, mb, TM, TN))   # X = a·X + BX
