"""
ancora/kernels/moe.py — grouped (segmented) MoE kernels: ONE launch per stage over ALL
experts, on PREALLOCATED buffers. Replaces moe_layer.MoEFFN's per-expert Python loop over
self-allocating linear_bf16 calls (which churns ~110 GPU buffers/backward → the flaky 80%
alloc-churn race, CLAUDE.md). Kills both the loop and the churn.

Design (variable-group, dropless — no capacity padding waste):
  1. Route on host (fp32, as MoEFFN) → topi (M,k), topw (M,k).
  2. build_layout: STABLE-sort the M*k (token,expert) assignments by expert, pad each expert's
     group up to a TM(128)-tile. Produces per-slot src_row/gate, per-tile expert id, and per-token
     slot indices. Stable sort + fixed per-tile accumulation ⇒ batch-invariant (CLAUDE.md MoE rule).
  3. _gather: Xg[slot] = h[src_row[slot]]  (pack tokens into expert-contiguous groups).
  4. _ggemm: grouped GEMM — each m-tile reads its expert id from tile_expert[] (data-dependent
     index, probed-OK) and indexes that expert's weight block. Used for gate, up, down.
  5. _swiglu_g: Ag = silu(Gg)*Ug on the grouped buffer.
  6. _combine: out[token] = Σ_j gate[slot_j] * Yg[slot_j]  (token gathers its k slots → no atomics).

Each output row of every GEMM/gather/combine depends only on its own input row + the expert
weights ⇒ a token's result is independent of where it lands in the group ⇒ batch-invariant.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cuda.tile as ct
from cuda.bindings import runtime as cudart, driver as cdrv
import ancora.env

from ancora.kernels.loss import GTM, GTN, GTK, _GpuArray, f32_to_bf16_bits as _f32bf
from ancora.kernels.fused import _acc_f32, _acc_f32_flat

TM, TN, TK = GTM, GTN, GTK      # 128, 128, 64 — same GEMM tiling as loss._gemm
SM, SN = 64, 128                # swiglu tile


# ── host: build the grouped layout from routing (stable → batch-invariant) ──────
def build_layout(topi: np.ndarray, topw: np.ndarray, E: int):
    """topi,topw: (M,k). Returns dict of int32/f32 host arrays for the grouped pipeline."""
    M, k = topi.shape
    tok = np.repeat(np.arange(M), k)                 # (M*k,) source token of each assignment
    exp = topi.reshape(-1).astype(np.int64)          # (M*k,) expert
    gat = topw.reshape(-1).astype(np.float32)        # (M*k,) gate weight
    sit = np.tile(np.arange(k), M)                   # which of the token's k slots
    order = np.argsort(exp, kind="stable")           # group by expert, STABLE (determinism)
    tok_s, gat_s, sit_s = tok[order], gat[order], sit[order]
    counts = np.bincount(exp[order], minlength=E)
    c = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)   # slice bounds in sorted arrays
    p = ((counts + TM - 1) // TM) * TM               # per-expert padded row count
    off = np.concatenate([[0], np.cumsum(p)]).astype(np.int64)      # grouped row offset per expert
    R = int(off[-1]); Rt = R // TM

    src_row   = np.zeros(R, np.int32)                # padding rows -> token 0 (masked out by combine)
    slot_gate = np.zeros(R, np.float32)
    tile_expert = np.zeros(max(Rt, 1), np.int32)
    tok_slots = np.zeros((M, k), np.int32)           # token -> its k grouped slot indices
    for e in range(E):
        lo, hi = int(c[e]), int(c[e + 1]); n = hi - lo
        base = int(off[e])
        if n:
            src_row[base:base + n]   = tok_s[lo:hi]
            slot_gate[base:base + n] = gat_s[lo:hi]
            tok_slots[tok_s[lo:hi], sit_s[lo:hi]] = base + np.arange(n)
        tile_expert[off[e] // TM: off[e + 1] // TM] = e
    off_tiles = (off // TM).astype(np.int32)         # per-expert m-tile offsets (for weight-grad)
    return dict(R=R, Rt=Rt, src_row=src_row, slot_gate=slot_gate, tile_expert=tile_expert,
               tok_slots=tok_slots, off_tiles=off_tiles, M=M, k=k)


# ── kernels ─────────────────────────────────────────────────────────────────
@ct.kernel
def _gather(src_row, X, Xg, H: ct.Constant[int]):
    """Xg[r] = X[src_row[r]].  X,Xg: bf16 bits (uint16), one (1,H) row per slot. Grid (R,)."""
    r = ct.bid(0)
    s = ct.reshape(ct.load(src_row, index=(r, 0), shape=(1, 1)), ())
    ct.store(Xg, index=(r, 0), tile=ct.load(X, index=(s, 0), shape=(1, H)))


@ct.kernel
def _ggemm(Xg, W, tile_expert, Out, Kb: ct.Constant[int], NE: ct.Constant[int]):
    """Out[mi,ni] = Xg[mi-tile] @ W[e]  where e = tile_expert[mi]. W laid (E*Kdim, N) bf16 bits;
    expert e occupies row-blocks [e*Kb, (e+1)*Kb). Out f32. Grid (Rt, N//TN). Padding tiles (device
    -route fixed grid) carry e==NE → mma skipped (stores zeros, ignored); host route never hits it."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    acc = ct.zeros((TM, TN), ct.float32)
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xg, index=(mi, kk), shape=(TM, TK)), ct.bfloat16)
            wt = ct.bitcast(ct.load(W, index=(e * Kb + kk, ni), shape=(TK, TN)), ct.bfloat16)
            acc = ct.mma(xt, wt, acc)
    ct.store(Out, index=(mi, ni), tile=acc)


@ct.kernel
def _swiglu_g(Gg, Ug, Ag):
    """Ag = silu(Gg) * Ug.  Gg,Ug f32 → Ag bf16 bits (uint16). Grid (R//SM, Ie//SN)."""
    mi, ni = ct.bid(0), ct.bid(1)
    g = ct.load(Gg, index=(mi, ni), shape=(SM, SN))
    u = ct.load(Ug, index=(mi, ni), shape=(SM, SN))
    a = (g / (1.0 + ct.exp2(-g * 1.4426950408889634))) * u      # silu(g)*u  (sigmoid via exp2)
    ct.store(Ag, index=(mi, ni), tile=ct.bitcast(ct.astype(a, ct.bfloat16), ct.uint16))


@ct.kernel
def _combine(tok_slots, slot_gate, Yg, Out, K: ct.Constant[int], H: ct.Constant[int]):
    """Out[m] = Σ_{j<K} slot_gate[tok_slots[m,j]] * Yg[tok_slots[m,j]].  Token-parallel gather of
    the k expert outputs (no atomics; fixed j order → batch-invariant). Yg,Out f32. Grid (M,)."""
    m = ct.bid(0)
    acc = ct.zeros((1, H), ct.float32)
    for j in range(K):
        s = ct.reshape(ct.load(tok_slots, index=(m, j), shape=(1, 1)), ())
        g = ct.reshape(ct.load(slot_gate, index=(s, 0), shape=(1, 1)), ())
        acc = acc + g * ct.load(Yg, index=(s, 0), shape=(1, H))
    ct.store(Out, index=(m, 0), tile=acc)


@ct.kernel
def _combine_bf16(tok_slots, slot_gate, Yg, Out, K: ct.Constant[int], H: ct.Constant[int]):
    """Like _combine but stores bf16 bits (for the resident layer's fp32-residual add)."""
    m = ct.bid(0)
    acc = ct.zeros((1, H), ct.float32)
    for j in range(K):
        s = ct.reshape(ct.load(tok_slots, index=(m, j), shape=(1, 1)), ())
        g = ct.reshape(ct.load(slot_gate, index=(s, 0), shape=(1, 1)), ())
        acc = acc + g * ct.load(Yg, index=(s, 0), shape=(1, H))
    ct.store(Out, index=(m, 0), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


# ── MXFP8 grouped GEMMs (2026-06-12, the MoE-family MXFP8 forward port) ────────
# Same recipe as the Qwen3 family (linear._fwd_mxfp8_bf16): per-row-per-32 E8M0 activation
# quant (quant._quant_mxfp8) × per-32-along-K weight quant (_quant_mxfp8_w over the PACKED
# (E·K, N) expert weights — 32 | K so blocks never straddle experts) → ct.mma_scaled.
# Backward stays BF16 (straight-through wrt the quant — the QAT recipe); Gg/Ug/Ag stay
# materialized exactly as the bf16 path so the backward is untouched.

QB_ = 32              # MXFP8 scale block (== quant.B)
KSC = TK // QB_       # scale chunks per K-tile (TK=64 → 2)


@ct.kernel(occupancy=2)
def _ggemm_mx(Xq, Xs, Wq, Ws, tile_expert, Out, Kb: ct.Constant[int], NE: ct.Constant[int],
              TNb: ct.Constant[int]):
    """Grouped MXFP8 GEMM: Out(f32) = (Xq⊙Xs) @ (Wq[e]⊙Ws[e]), e = tile_expert[mi].
    Xq (R,K) fp8 + Xs (R,K//32) E8M0; Wq packed (E·K, N) fp8 + Ws (E·K//32, N) E8M0
    (expert e's scales at row-blocks [e·Kb·KSC, +Kb·KSC) — the same (e·Kb+kk, ni) tile
    index as Wq, just with a (KSC, TNb) shape). Grid (Rt, N//TNb); padding tiles skip."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    acc = ct.zeros((TM, TNb), ct.float32)
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xq, index=(mi, kk), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
            xs = ct.bitcast(ct.load(Xs, index=(mi, kk), shape=(TM, KSC)), ct.float8_e8m0fnu)
            wt = ct.bitcast(ct.load(Wq, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.float8_e4m3fn)
            ws = ct.bitcast(ct.load(Ws, index=(e * Kb + kk, ni), shape=(KSC, TNb)), ct.float8_e8m0fnu)
            acc = ct.mma_scaled(xt, xs, wt, ws, acc)
    ct.store(Out, index=(mi, ni), tile=acc)


@ct.kernel(occupancy=2)
def _ggemm_gus_mx(Xq, Xs, Wgq, Wgs, Wuq, Wus, tile_expert, Ag,
                  Kb: ct.Constant[int], NE: ct.Constant[int], TNb: ct.Constant[int]):
    """DECODE-fused MXFP8 grouped gate+up+SwiGLU (the _ggemm_gus of the MXFP8 path): dual
    mma_scaled accumulators, SwiGLU epilogue VERBATIM, Ag bf16 bits out. Must be probed
    BITWISE vs _ggemm_mx(gate)+_ggemm_mx(up)+_swiglu_g (the training path) — the in-register
    accs equal the stored-f32 Gg/Ug losslessly. Grid (Rt, Ie//TNb)."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    ag = ct.zeros((TM, TNb), ct.float32)
    au = ct.zeros((TM, TNb), ct.float32)
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xq, index=(mi, kk), shape=(TM, TK), latency=10), ct.float8_e4m3fn)
            xs = ct.bitcast(ct.load(Xs, index=(mi, kk), shape=(TM, KSC)), ct.float8_e8m0fnu)
            wgt = ct.bitcast(ct.load(Wgq, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.float8_e4m3fn)
            wgs = ct.bitcast(ct.load(Wgs, index=(e * Kb + kk, ni), shape=(KSC, TNb)), ct.float8_e8m0fnu)
            wut = ct.bitcast(ct.load(Wuq, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.float8_e4m3fn)
            wus = ct.bitcast(ct.load(Wus, index=(e * Kb + kk, ni), shape=(KSC, TNb)), ct.float8_e8m0fnu)
            ag = ct.mma_scaled(xt, xs, wgt, wgs, ag)
            au = ct.mma_scaled(xt, xs, wut, wus, au)
    a = (ag / (1.0 + ct.exp2(-ag * 1.4426950408889634))) * au      # silu(g)*u — verbatim _swiglu_g
    ct.store(Ag, index=(mi, ni), tile=ct.bitcast(ct.astype(a, ct.bfloat16), ct.uint16))


# ── DECODE-MEGAKERNEL fused forward kernels (2026-06-11, rollout engine only) ──
# The Md=128 decode FFN is latency/launch-bound; these fold the forward chain into 3 launches
# (gate+up+swiglu → down → combine+residual) while staying BITWISE-equal to the separate
# training path: the mma accumulation is hardware (no contraction freedom), the SwiGLU
# expression is copied VERBATIM from _swiglu_g (mul/div/exp2 only — no a·b+c shape, so the
# recompile cannot re-contract it), and the bf16 round of the combine accumulator equals
# _combine_bf16's store + _residual_add_rf32's reload. Training keeps the separate kernels
# (backward needs Gg/Ug materialized); the decode↔training bitwise gates verify equality.

@ct.kernel(occupancy=2)
def _ggemm_gus(Xg, Wg, Wu, tile_expert, Ag, Kb: ct.Constant[int], NE: ct.Constant[int],
               TNb: ct.Constant[int]):
    """FUSED grouped gate+up GEMM + SwiGLU: Ag = bf16(silu(Xg@Wg[e]) * (Xg@Wu[e])).
    Dual register accumulators (Xg reused SAME orientation — safe); Gg/Ug never hit HBM.
    Grid (Rt, Ie//TNb). Padding tiles (e==NE) skip the mma → silu(0)*0 = 0 (== separate path)."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    ag = ct.zeros((TM, TNb), ct.float32)
    au = ct.zeros((TM, TNb), ct.float32)
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xg, index=(mi, kk), shape=(TM, TK), latency=10), ct.bfloat16)
            wgt = ct.bitcast(ct.load(Wg, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.bfloat16)
            wut = ct.bitcast(ct.load(Wu, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.bfloat16)
            ag = ct.mma(xt, wgt, ag)
            au = ct.mma(xt, wut, au)
    a = (ag / (1.0 + ct.exp2(-ag * 1.4426950408889634))) * au      # silu(g)*u — verbatim _swiglu_g
    ct.store(Ag, index=(mi, ni), tile=ct.bitcast(ct.astype(a, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=2)
def _ggemm_b(Xg, W, tile_expert, Out, Kb: ct.Constant[int], NE: ct.Constant[int],
             TNb: ct.Constant[int]):
    """_ggemm with a launch-time column tile (decode wants TNb=64: more blocks at Md=128)."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    acc = ct.zeros((TM, TNb), ct.float32)
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xg, index=(mi, kk), shape=(TM, TK), latency=10), ct.bfloat16)
            wt = ct.bitcast(ct.load(W, index=(e * Kb + kk, ni), shape=(TK, TNb), latency=10), ct.bfloat16)
            acc = ct.mma(xt, wt, acc)
    ct.store(Out, index=(mi, ni), tile=acc)


@ct.kernel
def _combine_rf32(tok_slots, slot_gate, Yg, R, Out, K: ct.Constant[int], H: ct.Constant[int]):
    """FUSED _combine_bf16 + _residual_add_rf32: Out(f32) = R(f32) + f32(bf16(Σ_j gate·Yg)).
    Same fixed j-order accumulation; the bf16 round equals the separate store+reload. Grid (M,)."""
    m = ct.bid(0)
    acc = ct.zeros((1, H), ct.float32)
    for j in range(K):
        s = ct.reshape(ct.load(tok_slots, index=(m, j), shape=(1, 1)), ())
        g = ct.reshape(ct.load(slot_gate, index=(s, 0), shape=(1, 1)), ())
        acc = acc + g * ct.load(Yg, index=(s, 0), shape=(1, H))
    res = ct.load(R, index=(m, 0), shape=(1, H))
    ct.store(Out, index=(m, 0), tile=res + ct.astype(ct.astype(acc, ct.bfloat16), ct.float32))


# ── backward kernels ──────────────────────────────────────────────────────────
@ct.kernel
def _combine_bwd(src_row, slot_gate, dOut, Yg, dYg, dsg, H: ct.Constant[int]):
    """Per-slot: dYg[r] = slot_gate[r]·dOut[src_row[r]] (bf16 bits);  dsg[r] = <dOut[tok], Yg[r]>
    (f32, for the gate-weight/router grad). Padding slots have gate=0 → dYg=0. Grid (R,)."""
    r = ct.bid(0)
    s = ct.reshape(ct.load(src_row,   index=(r, 0), shape=(1, 1)), ())
    g = ct.reshape(ct.load(slot_gate, index=(r, 0), shape=(1, 1)), ())
    do = ct.load(dOut, index=(s, 0), shape=(1, H))
    yg = ct.load(Yg,   index=(r, 0), shape=(1, H))
    ct.store(dYg, index=(r, 0), tile=ct.bitcast(ct.astype(g * do, ct.bfloat16), ct.uint16))
    ct.store(dsg, index=(r, 0), tile=ct.sum(do * yg, axis=-1, keepdims=True))


@ct.kernel
def _ggemm_acc(Xg, W, tile_expert, Out, Kb: ct.Constant[int], NE: ct.Constant[int]):
    """Like _ggemm but Out += Xg@W[e] (accumulate, for summing the gate+up data-gradients). Padding
    tiles (e==NE) skip the mma → Out unchanged (it was the zero _ggemm wrote for that padding tile)."""
    mi, ni = ct.bid(0), ct.bid(1)
    e = ct.reshape(ct.load(tile_expert, index=(mi, 0), shape=(1, 1)), ())
    acc = ct.load(Out, index=(mi, ni), shape=(TM, TN))
    if e < NE:
        for kk in range(Kb):
            xt = ct.bitcast(ct.load(Xg, index=(mi, kk), shape=(TM, TK)), ct.bfloat16)
            wt = ct.bitcast(ct.load(W, index=(e * Kb + kk, ni), shape=(TK, TN)), ct.bfloat16)
            acc = ct.mma(xt, wt, acc)
    ct.store(Out, index=(mi, ni), tile=acc)


@ct.kernel
def _swiglu_g_bwd(dAg, Gg, Ug, dGg, dUg):
    """SwiGLU backward on the grouped buffer. dGg,dUg out as bf16 bits. Grid (R//SM, Ie//SN)."""
    mi, ni = ct.bid(0), ct.bid(1)
    da = ct.load(dAg, index=(mi, ni), shape=(SM, SN))
    g  = ct.load(Gg,  index=(mi, ni), shape=(SM, SN))
    u  = ct.load(Ug,  index=(mi, ni), shape=(SM, SN))
    sig = 1.0 / (1.0 + ct.exp2(-g * 1.4426950408889634))
    s = g * sig                                                   # silu(g)
    dUgt = da * s
    dGgt = da * u * (sig * (1.0 + g * (1.0 - sig)))               # da · u · silu'(g)
    ct.store(dGg, index=(mi, ni), tile=ct.bitcast(ct.astype(dGgt, ct.bfloat16), ct.uint16))
    ct.store(dUg, index=(mi, ni), tile=ct.bitcast(ct.astype(dUgt, ct.bfloat16), ct.uint16))


@ct.kernel
def _ggemm_dw(Xg, dY, off_tile, dW, KoT: ct.Constant[int], MaxT: ct.Constant[int]):
    """Per-expert weight grad: dW[e] = Σ_{slots of e} Xg[slot]^T @ dY[slot].  Block (e, ki, ni)
    loops expert e's m-tiles [off_tile[e], off_tile[e+1]) (≤ MaxT, static unroll + runtime guard)
    accumulating in ONE block (no atomics → batch-invariant). dW laid (E*Ko, N): expert e at
    row-blocks [e*KoT, +KoT). Grid (E, Ko//TM, N//TN)."""
    e, ki, ni = ct.bid(0), ct.bid(1), ct.bid(2)
    lo = ct.reshape(ct.load(off_tile, index=(e, 0),     shape=(1, 1)), ())
    hi = ct.reshape(ct.load(off_tile, index=(e + 1, 0), shape=(1, 1)), ())
    acc = ct.zeros((TM, TN), ct.float32)
    for t in range(MaxT):
        mt = lo + t
        if mt < hi:
            xt  = ct.bitcast(ct.load(Xg, index=(mt, ki), shape=(TM, TM)), ct.bfloat16)
            dyt = ct.bitcast(ct.load(dY, index=(mt, ni), shape=(TM, TN)), ct.bfloat16)
            acc = ct.mma(ct.transpose(xt), dyt, acc)
    ct.store(dW, index=(e * KoT + ki, ni), tile=acc)


@ct.kernel
def _ggemm_dw_acc(Xg, dY, off_tile, dW, KoT: ct.Constant[int], MaxT: ct.Constant[int]):
    """_ggemm_dw accumulating onto the existing dW (GRADIENT ACCUMULATION, micro-batch ≥1).
    Experts with no tiles this micro-batch add 0 → dW unchanged. Grid (E, Ko//TM, N//TN)."""
    e, ki, ni = ct.bid(0), ct.bid(1), ct.bid(2)
    lo = ct.reshape(ct.load(off_tile, index=(e, 0),     shape=(1, 1)), ())
    hi = ct.reshape(ct.load(off_tile, index=(e + 1, 0), shape=(1, 1)), ())
    acc = ct.zeros((TM, TN), ct.float32)
    for t in range(MaxT):
        mt = lo + t
        if mt < hi:
            xt  = ct.bitcast(ct.load(Xg, index=(mt, ki), shape=(TM, TM)), ct.bfloat16)
            dyt = ct.bitcast(ct.load(dY, index=(mt, ni), shape=(TM, TN)), ct.bfloat16)
            acc = ct.mma(ct.transpose(xt), dyt, acc)
    base = ct.load(dW, index=(e * KoT + ki, ni), shape=(TM, TN))
    ct.store(dW, index=(e * KoT + ki, ni), tile=base + acc)


@ct.kernel
def _scatter_dh(tok_slots, dXg, dH, K: ct.Constant[int], H: ct.Constant[int]):
    """dH[token] = Σ_{j<K} dXg[tok_slots[token,j]].  Token-parallel (token's h fed K experts).
    No atomics → batch-invariant. dXg,dH f32. Grid (M,)."""
    m = ct.bid(0)
    acc = ct.zeros((1, H), ct.float32)
    for j in range(K):
        s = ct.reshape(ct.load(tok_slots, index=(m, j), shape=(1, 1)), ())
        acc = acc + ct.load(dXg, index=(s, 0), shape=(1, H))
    ct.store(dH, index=(m, 0), tile=acc)


# ── resident-variant backward kernels (bf16-bits I/O, for ResidentMoELayer) ──
@ct.kernel
def _combine_bwd_r(src_row, slot_gate, dOut, Yg, dYg, dsg, H: ct.Constant[int]):
    """_combine_bwd but dOut is bf16 BITS (the resident layer's grad). dYg bf16 bits, dsg f32."""
    r = ct.bid(0)
    s = ct.reshape(ct.load(src_row,   index=(r, 0), shape=(1, 1)), ())
    g = ct.reshape(ct.load(slot_gate, index=(r, 0), shape=(1, 1)), ())
    do = ct.astype(ct.bitcast(ct.load(dOut, index=(s, 0), shape=(1, H)), ct.bfloat16), ct.float32)
    yg = ct.load(Yg, index=(r, 0), shape=(1, H))
    ct.store(dYg, index=(r, 0), tile=ct.bitcast(ct.astype(g * do, ct.bfloat16), ct.uint16))
    ct.store(dsg, index=(r, 0), tile=ct.sum(do * yg, axis=-1, keepdims=True))


@ct.kernel
def _scatter_dh_bf16(tok_slots, dXg, dH, K: ct.Constant[int], H: ct.Constant[int]):
    """_scatter_dh but stores bf16 bits (grad wrt gh2 for the resident rms-backward)."""
    m = ct.bid(0)
    acc = ct.zeros((1, H), ct.float32)
    for j in range(K):
        s = ct.reshape(ct.load(tok_slots, index=(m, j), shape=(1, 1)), ())
        acc = acc + ct.load(dXg, index=(s, 0), shape=(1, H))
    ct.store(dH, index=(m, 0), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel
def _add2_bf16(A, B, C, H: ct.Constant[int]):
    """C = A + B, all bf16 bits (adds the router-path grad onto the expert-path grad). Grid (M,)."""
    m = ct.bid(0)
    a = ct.astype(ct.bitcast(ct.load(A, index=(m, 0), shape=(1, H)), ct.bfloat16), ct.float32)
    b = ct.astype(ct.bitcast(ct.load(B, index=(m, 0), shape=(1, H)), ct.bfloat16), ct.float32)
    ct.store(C, index=(m, 0), tile=ct.bitcast(ct.astype(a + b, ct.bfloat16), ct.uint16))


@ct.kernel
def _transpose_e(W, Wt, P: ct.Constant[int], Q: ct.Constant[int], T: ct.Constant[int]):
    """Per-expert weight transpose (bf16 bits): Wt[e] = W[e]ᵀ. W laid (E*P, Q), Wt laid (E*Q, P);
    expert e is row-block [e*P,+P) in W and [e*Q,+Q) in Wt. Refreshes the backward's transposed
    weights from the forward weights after a device-AdamW step (no host re-pack). Grid (E, P//T, Q//T)."""
    e, pi, qi = ct.bid(0), ct.bid(1), ct.bid(2)
    t = ct.bitcast(ct.load(W, index=(e * (P // T) + pi, qi), shape=(T, T)), ct.bfloat16)
    ct.store(Wt, index=(e * (Q // T) + qi, pi), tile=ct.bitcast(ct.transpose(t), ct.uint16))


class _View:
    """A reshaped, retyped view over an existing device buffer (shares the pointer, no copy).
    Lets the AdamW kernel treat the packed (E*·,·) weight / grad as a flat (R, 64) tile."""
    def __init__(self, base, shape, dtype):
        self._ptr = base._ptr
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": np.dtype(dtype).str,
                                         "data": (int(base._ptr), False), "version": 3}


# ── host helper: weight layout (E,*,*) -> (E*Kdim, N) bf16 bits, uploaded once ──
def pack_expert_weights(W_e: np.ndarray) -> np.ndarray:
    """(E, Kdim, N) f32 -> (E*Kdim, N) uint16 (bf16 bits) for _ggemm."""
    E, Kd, N = W_e.shape
    return _f32bf(W_e.reshape(E * Kd, N))


def _up(buf: _GpuArray, host: np.ndarray, si: int):
    """Upload host array into the front of a preallocated device buffer, on stream si
    (same stream as the consuming kernels — cross-stream upload races, CLAUDE.md)."""
    h = np.ascontiguousarray(host)
    cdrv.cuMemcpyHtoDAsync(buf._ptr, h, h.nbytes, si)
    return h                                          # keep alive until the caller syncs


class ExpertMuonScratch:
    """SHARED batched-NS scratch for Muon over the MoE experts (E SQUARE (M,M) experts packed
    (E*M, M)). Shared across ALL MoE layers AND the 3 weight tensors (Wg/Wu/Wd) — they NS
    sequentially on si, so one scratch serves all. ~167 MB one-time (E=16,M=1024); per-layer
    scratch would be 167 MB × N_moe and erase the saving. The only PER-TENSOR persistent Muon
    state is the momentum buffer (in GroupedMoEFFN._muon[name]['buf'])."""
    def __init__(self, E, M):
        Z = lambda *s: _GpuArray.zeros(s, np.uint16)
        self.gA, self.gA2, self.gB, self.gBX = Z(E * M, M), Z(E * M, M), Z(E * M, M), Z(E * M, M)
        self.u = Z(E * M, M)                          # momentum-Nesterov input → orthogonalized output
        self.recip = _GpuArray.zeros((E, 1), np.float32)

    def nbytes(self):
        return self.gA._nbytes * 5 + self.recip._nbytes

    def free(self):
        for b in (self.gA, self.gA2, self.gB, self.gBX, self.u, self.recip):
            b.free()


class GroupedMoEFFN:
    """Device-resident grouped MoE FFN — DROP-IN for moe_layer.MoEFFN (routes internally, fwd+bwd),
    but one launch per stage over ALL experts on PREALLOCATED buffers (no per-expert Python loop,
    no alloc/free churn → deterministic). Takes MoEFFN's `w` dict; returns the same grad structure."""
    def __init__(self, w, top_k, si=None, norm_topk=True, device_route=False, mxfp8=False,
                 optimizer="adamw", muon_scratch_e=None, muon_lr=0.02):
        g = w["gate_proj"]
        self.E, self.H, self.Ie = g.shape[0], g.shape[1], g.shape[2]
        self.w, self.k, self.si, self.norm_topk = w, top_k, si, norm_topk   # w is the MASTER dict (shared)
        # optimizer="muon": the 3 expert tensors (Wg/Wu/Wd, E square (H,Ie) matrices) use the BATCHED
        # resident Muon (one momentum buffer, no v) sharing muon_scratch_e; the router stays AdamW
        # (device/host). Drops the experts' v (≈ the FFN's bulk optimizer memory). Default "adamw".
        self.optimizer, self._muon_scr, self._muon_lr = optimizer, muon_scratch_e, muon_lr
        self.device_route = device_route        # True → gate+dispatch on device (no host round-trip in fwd)
        self._M = None; self._packed = False    # lazy pack on first forward; re-pack after weight update
        self.gacc = False                       # gradient accumulation: micro-batch ≥1 ADDS dW in place
        self.mxfp8 = mxfp8                      # MXFP8 forward GEMMs (backward stays BF16 — QAT recipe)
        self._wq_dirty_mx = True                # (re)quantize the packed expert weights before a forward
        self._mx_master = None                  # decode aliases the TRAINER's quant buffers + dirty flag

    def _pack(self, si):
        """(Re)pack the master weights into the device GEMM buffers. Called only when dirty (first
        forward / after an optimizer step sets self._packed=False). Buffers are ALLOCATED ONCE and
        RE-UPLOADED on re-pack (no per-step alloc/free → no leak, no churn)."""
        g, u, d = self.w["gate_proj"], self.w["up_proj"], self.w["down_proj"]
        packs = {"Wg": pack_expert_weights(g), "Wu": pack_expert_weights(u), "Wd": pack_expert_weights(d),
                 "WdT": pack_expert_weights(d.transpose(0, 2, 1)),   # downᵀ (E*H, Ie)  for d_Ag
                 "WgT": pack_expert_weights(g.transpose(0, 2, 1)),   # gateᵀ (E*Ie, H)  for d_Xg
                 "WuT": pack_expert_weights(u.transpose(0, 2, 1))}   # upᵀ
        if not hasattr(self, "Wg"):
            for name, arr in packs.items(): setattr(self, name, _GpuArray(arr))
        else:
            self._packkeep = [_up(getattr(self, name), arr, si) for name, arr in packs.items()]
        self._packed = True

    def _ensure_mx(self):
        """Allocate the MXFP8 buffers for the PACKED expert weights (fp8 + E8M0, ~17 MB each)."""
        if hasattr(self, "Wgq"):
            return
        E, H, Ie = self.E, self.H, self.Ie
        Z = lambda *s: _GpuArray.zeros(s, np.uint8)
        self.Wgq, self.Wgs = Z(E * H, Ie), Z(E * H // QB_, Ie)
        self.Wuq, self.Wus = Z(E * H, Ie), Z(E * H // QB_, Ie)
        self.Wdq, self.Wds = Z(E * Ie, H), Z(E * Ie // QB_, H)

    def _quant_w_mx(self, si):
        """(Re)quantize the packed bf16 expert weights → MXFP8 on device (same _quant_mxfp8_w
        colblock as the Qwen3 family; 32 | K so blocks never straddle experts)."""
        from ancora.kernels.quant import _quant_mxfp8_w, QWN
        self._ensure_mx()
        E, H, Ie = self.E, self.H, self.Ie
        ct.launch(si, (E * H // QB_, Ie // QWN, 1), _quant_mxfp8_w, (self.Wg, self.Wgq, self.Wgs))
        ct.launch(si, (E * H // QB_, Ie // QWN, 1), _quant_mxfp8_w, (self.Wu, self.Wuq, self.Wus))
        ct.launch(si, (E * Ie // QB_, H // QWN, 1), _quant_mxfp8_w, (self.Wd, self.Wdq, self.Wds))
        self._wq_dirty_mx = False

    def _mx_ready(self, si):
        """Quantize via the weight owner (the trainer when decode aliases its buffers)."""
        mm = self._mx_master or self
        if mm._wq_dirty_mx:
            mm._quant_w_mx(si)

    def _route(self, h):
        logits = h.astype(np.float32) @ self.w["router"].astype(np.float32)   # (M,E) fp32, read fresh
        z = logits - logits.max(1, keepdims=True)
        probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
        topi = np.argsort(-probs, axis=1, kind="stable")[:, :self.k]
        topw = np.take_along_axis(probs, topi, 1)
        if self.norm_topk: topw = topw / topw.sum(1, keepdims=True)
        return probs, topi, topw

    def _prealloc(self, M):
        H, Ie, E, k = self.H, self.Ie, self.E, self.k
        Rmax = M * k + E * TM
        self._M, self._Rmax = M, Rmax
        Z = lambda *s, dt=np.float32: _GpuArray.zeros(s, dt)
        self.Xg, self.Ag = Z(Rmax, H, dt=np.uint16), Z(Rmax, Ie, dt=np.uint16)
        self.Gg, self.Ug, self.Yg = Z(Rmax, Ie), Z(Rmax, Ie), Z(Rmax, H)
        self.Out, self.hbits = Z(M, H), Z(M, H, dt=np.uint16)
        self.gsrc, self.ggate = Z(Rmax, 1, dt=np.int32), Z(Rmax, 1)
        self.gtile = Z(Rmax // TM, 1, dt=np.int32); self.gtoks = Z(M, k, dt=np.int32)
        self.goff = Z(E + 1, 1, dt=np.int32)
        # backward scratch
        self.dOut = Z(M, H); self.dYg = Z(Rmax, H, dt=np.uint16); self.dsg = Z(Rmax, 1)
        self.dAg = Z(Rmax, Ie); self.dGg, self.dUg = Z(Rmax, Ie, dt=np.uint16), Z(Rmax, Ie, dt=np.uint16)
        self.dXg, self.dH = Z(Rmax, H), Z(M, H)
        self.dWd = Z(E * Ie, H); self.dWg, self.dWu = Z(E * H, Ie), Z(E * H, Ie)
        self.gdh2_e = Z(M, H, dt=np.uint16); self.dhr = Z(M, H, dt=np.uint16)   # resident: expert + router paths
        if self.mxfp8:                     # MXFP8 activation-quant scratch (grouped rows)
            self.Xq, self.Xss = Z(Rmax, H, dt=np.uint8), Z(Rmax, H // QB_, dt=np.uint8)
            self.Aq, self.Ass = Z(Rmax, Ie, dt=np.uint8), Z(Rmax, Ie // QB_, dt=np.uint8)
        if self.device_route:              # device gating: router weight (fp32) + topi/topw/probs on device
            self.Wr_dev = Z(self.H, self.E); self.dtopi = Z(M, self.k, dt=np.int32)
            self.dtopw = Z(M, self.k); self.dprobs = Z(M, self.E)
            self.dlogits = Z(M, self.E); self.G_router_dev = Z(self.H, self.E)   # router bwd (device)
            self.Gr_acc = Z(self.H, self.E)    # grad-accum mirror (router_dW always overwrites G_router_dev);
                                               # _acc_f32_flat grid scales as H·E//128 blocks
            from ancora.kernels.moe_dispatch import ROUTER_DW_NSPL
            self.G_router_part = Z(ROUTER_DW_NSPL * self.H, self.E)              # 2-pass dW split-M scratch

    # ── device-resident AdamW over the 3 expert weight tensors + host AdamW for the router ──
    def init_adamw(self, si, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        """Set up device AdamW: the packed bf16 forward weights Wg/Wu/Wd ARE the optimizer's p16
        (updated in place — no re-pack), with fp32 master + m,v on device. The transposed backward
        weights (WgT/WuT/WdT) are refreshed from the updated forward weights each step via _transpose_e
        (device, no host). Router AdamW: DEVICE (Wr_dev = the fp32 master the gating reads, updated in
        place) when device_route, else HOST. Call once after construction."""
        from ancora.optim.adamw import _pick_otm, C as AC
        if not self._packed: self._pack(si)
        self._AC, self._betas, self._eps, self._wd, self._t = AC, betas, eps, wd, 0
        E, H, Ie = self.E, self.H, self.Ie
        self._eopt = {}
        if self.optimizer == "muon":
            # batched-Muon expert state: fp32 master + ONE momentum buffer (NO m/v) per tensor; the
            # packed bf16 Wg/Wu/Wd ARE the p16 (refreshed by _cast after each NS update). Experts are
            # uniform-square (Ie==H) so the (E*M,N) master reshape is contiguous and M==N (no transpose).
            assert Ie == H, f"expert-Muon assumes uniform-square experts (Ie={Ie}==H={H})"
            M, N = H, Ie
            if self._muon_scr is None:
                self._muon_scr = ExpertMuonScratch(E, M); self._own_muon_scr = True
            self._muon = {}
            for name, master in (("Wg", self.w["gate_proj"]), ("Wu", self.w["up_proj"]), ("Wd", self.w["down_proj"])):
                self._muon[name] = dict(p32=_GpuArray(master.astype(np.float32).reshape(E * M, N).copy()),
                                        buf=_GpuArray.zeros((E * M, N), np.float32),
                                        p16=getattr(self, name))     # the packed bf16 weight (E*M,N)
        else:
            for name, master in (("Wg", self.w["gate_proj"]), ("Wu", self.w["up_proj"]), ("Wd", self.w["down_proj"])):
                flat = master.astype(np.float32).reshape(-1); R = flat.size // AC
                self._eopt[name] = dict(R=R, otm=_pick_otm(R),
                    p32=_GpuArray(flat.reshape(R, AC).copy()),
                    m=_GpuArray.zeros((R, AC), np.float32), v=_GpuArray.zeros((R, AC), np.float32),
                    p16=_View(getattr(self, name), (R, AC), np.uint16))
        if self.device_route:               # router DEVICE AdamW: Wr_dev (the fp32 weight the gating reads)
            Rr = (H * E) // AC               # IS the master, updated in place → no host readback, no re-upload
            self._ropt = dict(R=Rr, otm=_pick_otm(Rr),
                p32=_View(self.Wr_dev, (Rr, AC), np.float32),
                m=_GpuArray.zeros((Rr, AC), np.float32), v=_GpuArray.zeros((Rr, AC), np.float32),
                p16=_GpuArray.zeros((Rr, AC), np.uint16))      # dummy p16 (gating reads the fp32 p32=Wr_dev)
            self._router_dev_adam = True
        else:
            self._rm = np.zeros_like(self.w["router"], np.float32)   # router host AdamW moments
            self._rv = np.zeros_like(self.w["router"], np.float32)

    def step(self, si, lr=1e-3, muon_lr=None):
        """One optimizer step: update Wg/Wu/Wd (in place) from dWg/dWu/dWd, refresh the transposed
        weights, then the router (DEVICE AdamW on Wr_dev if device_route — fully sync-free; else HOST).
        backward(_resident) ran on si and step reads the grads on si (same-stream order) → no sync.
        optimizer="muon": the experts use the batched resident Muon (lr=muon_lr); the router stays AdamW."""
        from ancora.optim.adamw import _adamw
        self._t += 1; b1, b2 = self._betas; AC, eps, wd = self._AC, self._eps, self._wd
        ibc1 = 1.0 / (1.0 - b1 ** self._t); ibc2 = 1.0 / (1.0 - b2 ** self._t)
        grads = {"Wg": self.dWg, "Wu": self.dWu, "Wd": self.dWd}
        if self.optimizer == "muon":
            from ancora.kernels.muon_ns import _e_muon_mom, _e_muon_update, newton_schulz_resident_e, NTM, NTN
            from ancora.kernels.fused import _cast_bf16, RTM, RTN
            E, M, N = self.E, self.H, self.Ie; mb, nb = M // NTM, N // NTN; sc = self._muon_scr
            mlr = self._muon_lr if muon_lr is None else muon_lr      # square experts → lr_scale == lr
            for name, s in self._muon.items():
                g = _View(grads[name], (E * M, N), np.float32)
                ct.launch(si, (E, mb, nb), _e_muon_mom, (s["buf"], g, sc.u, 0.95, mb, NTM, NTN))
                newton_schulz_resident_e(sc.u, sc.gA, sc.gA2, sc.gB, sc.gBX, sc.recip, E, M, N, si)
                ct.launch(si, (E, mb, nb), _e_muon_update, (s["p32"], sc.u, float(mlr), mb, NTM, NTN))
                ct.launch(si, ((E * M) // RTM, N // RTN, 1), _cast_bf16, (s["p32"], s["p16"]))  # refresh bf16 Wx
        else:
            for name, s in self._eopt.items():
                R, otm = s["R"], s["otm"]
                g = _View(grads[name], (R, AC), np.float32)
                ct.launch(si, (R // otm, 1, 1), _adamw, (g, s["m"], s["v"], s["p32"], s["p16"], otm,
                          float(b1), float(b2), float(eps), float(lr), float(wd), float(ibc1), float(ibc2)))
        E, H, Ie, T = self.E, self.H, self.Ie, 128            # refresh transposed backward weights (device)
        ct.launch(si, (E, H // T, Ie // T), _transpose_e, (self.Wg, self.WgT, H, Ie, T))
        ct.launch(si, (E, H // T, Ie // T), _transpose_e, (self.Wu, self.WuT, H, Ie, T))
        ct.launch(si, (E, Ie // T, H // T), _transpose_e, (self.Wd, self.WdT, Ie, H, T))
        self._wq_dirty_mx = True          # AdamW changed Wg/Wu/Wd in place → re-quant before next fwd
        if getattr(self, "_router_dev_adam", False):          # router DEVICE AdamW (Wr_dev = the fp32 master)
            # restore the (possibly accumulated) router grad — Gr_acc mirrors G_router_dev exactly
            # when no accumulation happened (the micro-batch-0 copy), so this is bit-neutral then
            cdrv.cuMemcpyDtoDAsync(self.G_router_dev._ptr, self.Gr_acc._ptr, self.H * self.E * 4, si)
            s = self._ropt; gr = _View(self.G_router_dev, (s["R"], AC), np.float32)
            ct.launch(si, (s["R"] // s["otm"], 1, 1), _adamw, (gr, s["m"], s["v"], s["p32"], s["p16"], s["otm"],
                      float(b1), float(b2), float(eps), float(lr), 0.0, float(ibc1), float(ibc2)))
        else:                                                 # router HOST AdamW (host-route path)
            g = self.G_router
            self._rm = b1 * self._rm + (1 - b1) * g; self._rv = b2 * self._rv + (1 - b2) * g * g
            upd = (self._rm * ibc1) / (np.sqrt(self._rv * ibc2) + eps) + wd * self.w["router"]
            self.w["router"] = self.w["router"].astype(np.float32) - lr * upd

    def forward(self, h, stream_int=None):
        """h:(M,H) f32 → (out (M,H) f32, None). The None keeps MoEFFN's (out, cache) contract; the
        backward state is cached on self (each layer owns its own GroupedMoEFFN)."""
        M, H = h.shape; Ie, E, k = self.Ie, self.E, self.k
        si = stream_int if stream_int is not None else self.si
        if not self._packed: self._pack(si)
        if self._M != M: self._prealloc(M)
        probs, topi, topw = self._route(h)
        lay = build_layout(topi, topw, E); R, Rt = lay["R"], lay["Rt"]
        keep = [_up(self.hbits, _f32bf(h), si),
                _up(self.gsrc,  lay["src_row"].reshape(-1, 1), si),
                _up(self.ggate, lay["slot_gate"].reshape(-1, 1), si),
                _up(self.gtile, lay["tile_expert"].reshape(-1, 1), si),
                _up(self.gtoks, lay["tok_slots"], si),
                _up(self.goff,  lay["off_tiles"].reshape(-1, 1), si)]
        ct.launch(si, (R, 1, 1), _gather, (self.gsrc, self.hbits, self.Xg, H))
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.Xg, self.Wg, self.gtile, self.Gg, H // TK, E))
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.Xg, self.Wu, self.gtile, self.Ug, H // TK, E))
        ct.launch(si, (R // SM, Ie // SN, 1), _swiglu_g, (self.Gg, self.Ug, self.Ag))
        ct.launch(si, (Rt, H // TN, 1), _ggemm, (self.Ag, self.Wd, self.gtile, self.Yg, Ie // TK, E))
        ct.launch(si, (M, 1, 1), _combine, (self.gtoks, self.ggate, self.Yg, self.Out, k, H))
        cudart.cudaStreamSynchronize(si)
        self._h, self._probs, self._topi, self._lay, self._keep = h, probs, topi, lay, keep
        return self.Out.to_numpy(), None

    def forward_resident(self, gh2, gmlp, si):
        """Resident MoE FFN: gh2 (device (M,H) bf16 bits) → gmlp (device (M,H) bf16 bits). The ONLY
        host round-trip is the router (sync once, download gh2, route on host, upload the small layout);
        gather/grouped-GEMMs/swiglu/combine all chain on `si` with device buffers. Caller syncs."""
        M, H = gh2.shape; Ie, E, k = self.Ie, self.E, self.k
        if not self._packed: self._pack(si)
        if self._M != M: self._prealloc(M)
        if self.device_route:
            from ancora.kernels.moe_dispatch import router_gate, build_layout_dev
            Rmax = self._Rmax; Rtmax = Rmax // TM       # FIXED grid (R unknown on host; padding → 0)
            if not getattr(self, "_router_dev_adam", False):  # before device AdamW owns Wr_dev: seed from host master
                self._wrkeep = _up(self.Wr_dev, self.w["router"].astype(np.float32), si)
            router_gate(gh2, self.Wr_dev, self.dtopi, self.dtopw, self.dprobs, M, H, E, k, int(self.norm_topk), si)
            build_layout_dev(self.dtopi, self.dtopw, self.gsrc, self.ggate, self.gtile, self.gtoks, self.goff,
                             M, k, E, TM, Rmax, Rtmax, si)
            ct.launch(si, (Rmax, 1, 1), _gather, (self.gsrc, gh2, self.Xg, H))
            if self.mxfp8:               # MXFP8 expert GEMMs (Gg/Ug/Ag stay materialized for the BF16 bwd)
                from ancora.kernels.quant import _quant_mxfp8, QTM
                self._mx_ready(si)
                ct.launch(si, (Rmax // QTM, 1, 1), _quant_mxfp8, (self.Xg, self.Xq, self.Xss, H // QB_))
                ct.launch(si, (Rtmax, Ie // TN, 1), _ggemm_mx, (self.Xq, self.Xss, self.Wgq, self.Wgs, self.gtile, self.Gg, H // TK, E, TN))
                ct.launch(si, (Rtmax, Ie // TN, 1), _ggemm_mx, (self.Xq, self.Xss, self.Wuq, self.Wus, self.gtile, self.Ug, H // TK, E, TN))
                ct.launch(si, (Rmax // SM, Ie // SN, 1), _swiglu_g, (self.Gg, self.Ug, self.Ag))
                ct.launch(si, (Rmax // QTM, 1, 1), _quant_mxfp8, (self.Ag, self.Aq, self.Ass, Ie // QB_))
                ct.launch(si, (Rtmax, H // TN, 1), _ggemm_mx, (self.Aq, self.Ass, self.Wdq, self.Wds, self.gtile, self.Yg, Ie // TK, E, TN))
            else:
                ct.launch(si, (Rtmax, Ie // TN, 1), _ggemm, (self.Xg, self.Wg, self.gtile, self.Gg, H // TK, E))
                ct.launch(si, (Rtmax, Ie // TN, 1), _ggemm, (self.Xg, self.Wu, self.gtile, self.Ug, H // TK, E))
                ct.launch(si, (Rmax // SM, Ie // SN, 1), _swiglu_g, (self.Gg, self.Ug, self.Ag))
                ct.launch(si, (Rtmax, H // TN, 1), _ggemm, (self.Ag, self.Wd, self.gtile, self.Yg, Ie // TK, E))
            ct.launch(si, (M, 1, 1), _combine_bf16, (self.gtoks, self.ggate, self.Yg, gmlp, k, H))
            self._devroute, self._gh2 = True, gh2        # backward: fixed-grid + small routing download
            return                                       # NO host sync / download → forward fully resident
        cudart.cudaStreamSynchronize(si)                    # gh2 must be finished before we read it
        h = (gh2.to_numpy().astype(np.uint32) << 16).view(np.float32)   # bf16 bits → fp32 (router input)
        probs, topi, topw = self._route(h)
        lay = build_layout(topi, topw, E); R, Rt = lay["R"], lay["Rt"]
        keep = [_up(self.gsrc,  lay["src_row"].reshape(-1, 1), si),
                _up(self.ggate, lay["slot_gate"].reshape(-1, 1), si),
                _up(self.gtile, lay["tile_expert"].reshape(-1, 1), si),
                _up(self.gtoks, lay["tok_slots"], si),
                _up(self.goff,  lay["off_tiles"].reshape(-1, 1), si)]
        ct.launch(si, (R, 1, 1), _gather, (self.gsrc, gh2, self.Xg, H))         # gather from DEVICE gh2
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.Xg, self.Wg, self.gtile, self.Gg, H // TK, E))
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.Xg, self.Wu, self.gtile, self.Ug, H // TK, E))
        ct.launch(si, (R // SM, Ie // SN, 1), _swiglu_g, (self.Gg, self.Ug, self.Ag))
        ct.launch(si, (Rt, H // TN, 1), _ggemm, (self.Ag, self.Wd, self.gtile, self.Yg, Ie // TK, E))
        ct.launch(si, (M, 1, 1), _combine_bf16, (self.gtoks, self.ggate, self.Yg, gmlp, k, H))  # → DEVICE gmlp
        self._h, self._probs, self._topi, self._lay, self._keep, self._devroute = h, probs, topi, lay, keep, False

    def forward_resident_dec(self, gh2, gres, gout, si):
        """DECODE-FUSED resident MoE FFN (rollout engine only): gh2 (M,H) bf16 bits + gres (M,H)
        f32 residual → gout (M,H) f32 = gres + FFN(gh2), in 6 launches: router_gate →
        build_layout_dev → _gather → _ggemm_gus (gate+up+SwiGLU, Gg/Ug never hit HBM) →
        _ggemm_b (down, TN=64) → _combine_rf32 (combine + residual). BITWISE == forward_resident
        + _residual_add_rf32 (probed, tests/kernels/_probe_decode_fused.py). device_route only;
        no backward state (decode never trains)."""
        M, H = gh2.shape; Ie, E, k = self.Ie, self.E, self.k
        assert self.device_route, "decode-fused MoE FFN requires device_route"
        from ancora.kernels.moe_dispatch import router_gate, build_layout_dev
        Rmax = self._Rmax; Rtmax = Rmax // TM
        router_gate(gh2, self.Wr_dev, self.dtopi, self.dtopw, self.dprobs, M, H, E, k,
                    int(self.norm_topk), si)
        build_layout_dev(self.dtopi, self.dtopw, self.gsrc, self.ggate, self.gtile, self.gtoks,
                         self.goff, M, k, E, TM, Rmax, Rtmax, si)
        ct.launch(si, (Rmax, 1, 1), _gather, (self.gsrc, gh2, self.Xg, H))
        if self.mxfp8:                   # mirror the trainer's MXFP8 path (quant kernels/order identical)
            from ancora.kernels.quant import _quant_mxfp8, QTM
            self._mx_ready(si)           # quantized by the weight OWNER (trainer) → same bytes
            ct.launch(si, (Rmax // QTM, 1, 1), _quant_mxfp8, (self.Xg, self.Xq, self.Xss, H // QB_))
            ct.launch(si, (Rtmax, Ie // 32, 1), _ggemm_gus_mx,
                      (self.Xq, self.Xss, self.Wgq, self.Wgs, self.Wuq, self.Wus, self.gtile,
                       self.Ag, H // TK, E, 32))
            ct.launch(si, (Rmax // QTM, 1, 1), _quant_mxfp8, (self.Ag, self.Aq, self.Ass, Ie // QB_))
            ct.launch(si, (Rtmax, H // 64, 1), _ggemm_mx,
                      (self.Aq, self.Ass, self.Wdq, self.Wds, self.gtile, self.Yg, Ie // TK, E, 64))
        else:
            ct.launch(si, (Rtmax, Ie // 32, 1), _ggemm_gus,
                      (self.Xg, self.Wg, self.Wu, self.gtile, self.Ag, H // TK, E, 32))
            ct.launch(si, (Rtmax, H // 64, 1), _ggemm_b,
                      (self.Ag, self.Wd, self.gtile, self.Yg, Ie // TK, E, 64))
        ct.launch(si, (M, 1, 1), _combine_rf32, (self.gtoks, self.ggate, self.Yg, gres, gout, k, H))

    def backward_resident(self, gdout, gdh2, si):
        """Resident MoE FFN backward: gdout (device (M,H) bf16 bits, grad of gmlp) → gdh2 (device
        (M,H) bf16 bits, grad of gh2). Expert weight grads → device dWd/dWg/dWu (for AdamW). Router
        grad → host (self.G_router, self.G_h*ᵀ); its d_h path is added on-device. 2 round-trips:
        dsg download + d_h_router upload (router GEMM N=E=16 doesn't fit the tile)."""
        M, H = gdout.shape; Ie, E, k = self.Ie, self.E, self.k
        MaxT = (M + TM - 1) // TM
        if getattr(self, "_devroute", False):
            R, Rt = self._Rmax, self._Rmax // TM        # fixed grid (device layout; padding harmless)
        else:
            R, Rt = self._lay["R"], self._lay["Rt"]
        ct.launch(si, (R, 1, 1), _combine_bwd_r, (self.gsrc, self.ggate, gdout, self.Yg, self.dYg, self.dsg, H))
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.dYg, self.WdT, self.gtile, self.dAg, H // TK, E))
        ct.launch(si, (R // SM, Ie // SN, 1), _swiglu_g_bwd, (self.dAg, self.Gg, self.Ug, self.dGg, self.dUg))
        ct.launch(si, (Rt, H // TN, 1), _ggemm, (self.dGg, self.WgT, self.gtile, self.dXg, Ie // TK, E))
        ct.launch(si, (Rt, H // TN, 1), _ggemm_acc, (self.dUg, self.WuT, self.gtile, self.dXg, Ie // TK, E))
        ct.launch(si, (M, 1, 1), _scatter_dh_bf16, (self.gtoks, self.dXg, self.gdh2_e, k, H))   # expert path
        dwk = _ggemm_dw_acc if self.gacc else _ggemm_dw   # gradient accumulation: micro-batch ≥1 adds
        ct.launch(si, (E, Ie // TM, H // TN), dwk, (self.Ag, self.dYg, self.goff, self.dWd, Ie // TM, MaxT))
        ct.launch(si, (E, H // TM, Ie // TN), dwk, (self.Xg, self.dGg, self.goff, self.dWg, H // TM, MaxT))
        ct.launch(si, (E, H // TM, Ie // TN), dwk, (self.Xg, self.dUg, self.goff, self.dWu, H // TM, MaxT))
        # router grad — dsg → gate-weight grad → softmax/renorm bwd → d_logits → G_router (HᵀM·ME) +
        # d_h_router (ME·EH). device_route: ALL on device (gate_bwd → router_dW → router_dh, the last
        # fuses the expert+router add into gdh2) — NO host sync/download (G_router_dev read in step()).
        # host_route: the original host path (dsg/topi/probs on host).
        if getattr(self, "_devroute", False):
            from ancora.kernels.moe_dispatch import router_gate_bwd, router_dW, router_dh
            router_gate_bwd(self.dsg, self.gtoks, self.dtopi, self.dprobs, self.dlogits, M, E, k, int(self.norm_topk), si)
            router_dW(self._gh2, self.dlogits, self.G_router_dev, self.G_router_part, M, H, E, si)  # G_router (2-pass)
            # router-grad accumulation: the raw-CUDA router_dW always overwrites G_router_dev, so a
            # tiny (H,E) f32 accum mirror carries the sum (copy on micro-batch 0, add on ≥1); step()
            # copies it back before the router AdamW (identical bits when no accumulation happened).
            if self.gacc:
                ct.launch(si, (self.H * self.E // 128, 1, 1), _acc_f32_flat,
                          (_View(self.G_router_dev, (self.H * self.E, 1), np.float32),
                           _View(self.Gr_acc, (self.H * self.E, 1), np.float32)))
            else:
                cdrv.cuMemcpyDtoDAsync(self.Gr_acc._ptr, self.G_router_dev._ptr,
                                       self.H * self.E * 4, si)
            router_dh(self.dlogits, self.Wr_dev, self.gdh2_e, gdh2, M, H, E, si) # gdh2 = expert + router
            return
        cudart.cudaStreamSynchronize(si)
        dsg = self.dsg.to_numpy()[:, 0]
        topi, tok_slots, h = self._topi, self._lay["tok_slots"], self._h
        d_w = np.zeros((M, E), np.float32); d_w[np.arange(M)[:, None], topi] = dsg[tok_slots]
        d_logits = self._gate_backward(d_w)
        g_router = h.astype(np.float32).T @ d_logits                            # (H,E) router weight grad (host)
        self.G_router = self.G_router + g_router if self.gacc else g_router     # host-route grad accumulation
        d_h_router = (d_logits @ self.w["router"].astype(np.float32).T)         # (M,H) router → gh2 path
        self._bkeep = _up(self.dhr, _f32bf(d_h_router), si)                     # upload + add on-device
        ct.launch(si, (M, 1, 1), _add2_bf16, (self.gdh2_e, self.dhr, gdh2, H))

    def backward(self, d_out, h=None, cache=None, stream_int=None):
        """MoEFFN-compatible signature: h/cache are ignored (state is on self from forward).
        Returns (d_h (M,H), grads {router,gate_proj,up_proj,down_proj})."""
        M, H = d_out.shape; Ie, E, k = self.Ie, self.E, self.k
        si = stream_int if stream_int is not None else self.si
        lay = self._lay; R, Rt = lay["R"], lay["Rt"]; MaxT = (M + TM - 1) // TM
        keep = [_up(self.dOut, d_out.astype(np.float32), si)]
        ct.launch(si, (R, 1, 1), _combine_bwd, (self.gsrc, self.ggate, self.dOut, self.Yg, self.dYg, self.dsg, H))
        ct.launch(si, (Rt, Ie // TN, 1), _ggemm, (self.dYg, self.WdT, self.gtile, self.dAg, H // TK, E))   # d_Ag = dYg@downᵀ
        ct.launch(si, (R // SM, Ie // SN, 1), _swiglu_g_bwd, (self.dAg, self.Gg, self.Ug, self.dGg, self.dUg))
        ct.launch(si, (Rt, H // TN, 1), _ggemm, (self.dGg, self.WgT, self.gtile, self.dXg, Ie // TK, E))   # d_Xg = dGg@gateᵀ
        ct.launch(si, (Rt, H // TN, 1), _ggemm_acc, (self.dUg, self.WuT, self.gtile, self.dXg, Ie // TK, E))#       + dUg@upᵀ
        ct.launch(si, (M, 1, 1), _scatter_dh, (self.gtoks, self.dXg, self.dH, k, H))
        ct.launch(si, (E, Ie // TM, H // TN), _ggemm_dw, (self.Ag, self.dYg, self.goff, self.dWd, Ie // TM, MaxT))
        ct.launch(si, (E, H // TM, Ie // TN), _ggemm_dw, (self.Xg, self.dGg, self.goff, self.dWg, H // TM, MaxT))
        ct.launch(si, (E, H // TM, Ie // TN), _ggemm_dw, (self.Xg, self.dUg, self.goff, self.dWu, H // TM, MaxT))
        cudart.cudaStreamSynchronize(si)
        d_h = self.dH.to_numpy()
        g = {"down_proj": self.dWd.to_numpy().reshape(E, Ie, H),
             "gate_proj": self.dWg.to_numpy().reshape(E, H, Ie),
             "up_proj":   self.dWu.to_numpy().reshape(E, H, Ie)}
        # router grad (host): map per-slot dsg → per-(token,expert) gate-weight grad, then softmax/renorm bwd
        dsg = self.dsg.to_numpy()[:, 0]
        topi = self._topi
        d_w = np.zeros((M, E), np.float32)
        d_w[np.arange(M)[:, None], topi] = dsg[lay["tok_slots"]]           # only real slots (no padding)
        d_logits = self._gate_backward(d_w)
        g["router"] = self._h.astype(np.float32).T @ d_logits
        d_h = d_h + d_logits @ self.w["router"].astype(np.float32).T      # router path into d_h
        return d_h, g

    def _gate_backward(self, d_w):
        probs, topi = self._probs, self._topi; M, E = probs.shape
        d_sel = np.take_along_axis(d_w, topi, 1)
        if self.norm_topk:
            raw = np.take_along_axis(probs, topi, 1); sm = raw.sum(1, keepdims=True)
            dot = (d_sel * (raw / sm)).sum(1, keepdims=True)
            d_sel = (d_sel - dot) / sm
        d_probs = np.zeros((M, E), np.float32); np.put_along_axis(d_probs, topi, d_sel, 1)
        return probs * (d_probs - (d_probs * probs).sum(1, keepdims=True))
