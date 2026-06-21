"""
ancora/kernels/fused.py — plumbing kernels for the DEVICE-RESIDENT forward.

The device-resident layer chains kernels on-device with persistent buffers (no host
round-trips → ~100× vs the host-glue model, see [[mfu-strategy]]). For that, the dtype
seams between kernels must stay on-device:
  • _gemm_bf16  — GEMM that outputs BF16 bits directly (so gemm→{norm,swiglu,gemm} needs
                  no separate cast; the f32 accumulator is rounded to bf16 on store).
  • _residual_add — out = bf16(a + b), a,b BF16 bits (fuses the residual + f32→bf16 seam).
  • _cast_bf16  — f32 → BF16 bits (for the attention f32 output → o_proj seam).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import cuda.tile as ct
import ancora.env  # noqa: F401


# ── MEGAKERNEL fusion #1: gate/up MXFP8 GEMM + SwiGLU + MXFP8 quant in ONE kernel ──
# The separate path writes gate(M,I) + up(M,I) bf16 to HBM (96 MB) then re-reads both in
# swiglu_q (another 96 MB) — pure memory traffic the tensor cores stall on. Here both GEMMs
# (shared activation h, two weights Wg/Wu) accumulate in registers; the SwiGLU + per-32 E8M0
# quant run in the epilogue, so gate/up NEVER touch HBM. Output is the down_proj input
# directly: fp8(M,I) + scale(M,I//32). This is the CODA epilogue-fusion blueprint, step 1
# ([[coda-epilogue-fusion]], [[gemm-mfu-ceiling]]).
#
# N-tile = 32 = exactly one MXFP8 quant block → one E8M0 scale per output tile, giving the
# clean (M, I//32) scale layout the down GEMM expects (mma_scaled FP8 block is hardwired 32).
# h is reused non-transposed in both mma_scaled (SAME orientation → safe, CLAUDE pitfall 0a).

@ct.kernel(occupancy=2)
def _gateup_swiglu_q(h, hs, wg, wgs, wu, wus, fp8_out, scale_out,
                     KB: ct.Constant[int], TM_: ct.Constant[int], TK_: ct.Constant[int]):
    """fused gate/up MXFP8 GEMM → SwiGLU → MXFP8 quant. Grid (M//TM_, I//32).
    h(M,H) fp8 + hs(M,H//32) E8M0; wg/wu(H,I) fp8 + wgs/wus(H//32,I) E8M0 (weights
    pre-transposed (K,N)). → fp8_out(M,I) u8 + scale_out(M,I//32) u8."""
    TN = 32
    m, n = ct.bid(0), ct.bid(1)
    ag = ct.zeros((TM_, TN), ct.float32)
    au = ct.zeros((TM_, TN), ct.float32)
    KS = TK_ // 32
    for k in range(KB):
        th  = ct.bitcast(ct.load(h,  index=(m, k), shape=(TM_, TK_), latency=10), ct.float8_e4m3fn)
        ths = ct.bitcast(ct.load(hs, index=(m, k), shape=(TM_, KS)), ct.float8_e8m0fnu)
        twg  = ct.bitcast(ct.load(wg,  index=(k, n), shape=(TK_, TN), latency=10), ct.float8_e4m3fn)
        twgs = ct.bitcast(ct.load(wgs, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        twu  = ct.bitcast(ct.load(wu,  index=(k, n), shape=(TK_, TN), latency=10), ct.float8_e4m3fn)
        twus = ct.bitcast(ct.load(wus, index=(k, n), shape=(KS, TN)), ct.float8_e8m0fnu)
        ag = ct.mma_scaled(th, ths, twg, twgs, ag)
        au = ct.mma_scaled(th, ths, twu, twus, au)
    sig = 0.5 + 0.5 * ct.tanh(0.5 * ag)        # SwiGLU: silu(gate)*up
    y = (ag * sig) * au
    amax = ct.max(ct.maximum(y, 0.0 - y), axis=-1, keepdims=True)   # per-32 E8M0 quant
    ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
    byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((TM_, 1), 0, ct.uint32))
    sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
    fp8 = ct.bitcast(ct.astype(y / ct.broadcast_to(sc, (TM_, TN)), ct.float8_e4m3fn), ct.uint8)
    ct.store(fp8_out,   index=(m, n), tile=fp8)
    ct.store(scale_out, index=(m, n), tile=ct.astype(byte, ct.uint8))


@ct.kernel(occupancy=2)
def _gemm_bf16(A, B, C, KB: ct.Constant[int],
               TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = A @ B, BF16-bit inputs, f32 accumulate, BF16-bit OUTPUT. Fixed K-order → invariant."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    ct.store(C, index=(m, n), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


# ── DECODE-MEGAKERNEL GEMM fusions (2026-06-11) ───────────────────────────────
# The Md=128 decode step is GPU-bound on ~250 serial micro-kernels; the GEMM family is
# fusion-SAFE for the bitwise (ratio=1) requirement because the f32 accumulation lives in
# hardware mma instructions (no compiler FMA-contraction freedom) and the epilogue ops
# below replicate the separate kernels' math exactly:
#   _gemm_bf16_res : C(f32) = R(f32) + f32(bf16(A@B))  — folds _residual_add_rf32 into the
#                    GEMM epilogue (bf16 round of the accumulator == the separate kernel's
#                    bf16 store + f32 reload, lossless either way).
#   _gemm_af32_res : same, but A is native f32 and is RNE-rounded to bf16 in-register on
#                    load — exactly fused._cast_bf16 — so the attention-output f32→bf16
#                    seam + o_proj GEMM + residual add become ONE kernel.
# TN is a launch arg: at M=128 the decode shapes want TN=32 (N=1024 → 32 blocks instead of
# 8; measured 2× and BITWISE-invariant to TN, tests/kernels/_probe_decode_tiles.py).

@ct.kernel(occupancy=2)
def _gemm_bf16_res(A, B, R, C, KB: ct.Constant[int],
                   TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C(f32) = R(f32) + f32(bf16(A @ B)); A bf16 bits. == _gemm_bf16 then _residual_add_rf32."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    res = ct.load(R, index=(m, n), shape=(TM_, TN_))
    ct.store(C, index=(m, n), tile=res + ct.astype(ct.astype(acc, ct.bfloat16), ct.float32))


@ct.kernel(occupancy=2)
def _gemm_af32_res(A, B, R, C, KB: ct.Constant[int],
                   TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C(f32) = R(f32) + f32(bf16(bf16(A) @ B)); A native f32, RNE-rounded on load
    (== _cast_bf16 → _gemm_bf16 → _residual_add_rf32 in ONE kernel)."""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.astype(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, tb, acc)
    res = ct.load(R, index=(m, n), shape=(TM_, TN_))
    ct.store(C, index=(m, n), tile=res + ct.astype(ct.astype(acc, ct.bfloat16), ct.float32))


# ── backward GEMMs (our weight convention: forward y = x@W, W is (K,N)) ───────
# dx = dy @ Wᵀ  (dy(M,N), W(K,N) → dx(M,K) bf16) ; dW = xᵀ @ dy ((K,M)@(M,N) → dW(K,N) f32).
# Per-tile transpose (one orientation each → no reuse-both-orientations crash). dx in bf16
# (activation grad), dW in f32 (param grad → optimizer).

@ct.kernel(occupancy=4)
def _gemm_dx(dy, W, dx, NB: ct.Constant[int],
             TM_: ct.Constant[int], TK_: ct.Constant[int], TN_: ct.Constant[int]):
    """dx = dy @ Wᵀ.  Grid (M//TM_, K//TK_), reduce over N."""
    m, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TK_), ct.float32)
    for n in range(NB):
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(TM_, TN_), latency=10), ct.bfloat16)
        tw = ct.bitcast(ct.load(W, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(tdy, ct.transpose(tw), acc)            # (TM,TN)@(TN,TK)
    ct.store(dx, index=(m, k), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


# Stochastic-rounding variant of _gemm_dx: round the f32 activation-gradient accumulator to bf16
# with a per-element dither instead of round-to-nearest, so the fp32→bf16 downcast is UNBIASED
# (E[bf16]=f32). The recipe ([[precision-format-decision]]) mandates SR on every high→low gradient
# downcast. cuda-tile has no RNG, so the dither is an in-kernel counter hash of the GLOBAL output
# coords (m*TM_+row, k*TK_+col) XOR a per-step `seed` — coord-keyed ⇒ batch-invariant for a fixed
# seed (same token's gradient rounds identically regardless of batch), seed varies per step ⇒
# unbiased across steps. (Verified unbiased in tests/kernels/_probe_sr_ops.py.)

@ct.kernel(occupancy=4)
def _gemm_dx_sr(dy, W, dx, NB: ct.Constant[int],
                TM_: ct.Constant[int], TK_: ct.Constant[int], TN_: ct.Constant[int], seed):
    """dx = SR_bf16(dy @ Wᵀ).  Grid (M//TM_, K//TK_), reduce over N. = _gemm_dx + stochastic round."""
    m, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TK_), ct.float32)
    for n in range(NB):
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(TM_, TN_), latency=10), ct.bfloat16)
        tw = ct.bitcast(ct.load(W, index=(k, n), shape=(TK_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(tdy, ct.transpose(tw), acc)
    mu = ct.astype(m, ct.uint32); ku = ct.astype(k, ct.uint32); su = ct.astype(seed, ct.uint32)
    row = ct.broadcast_to(ct.reshape(ct.arange(TM_, dtype=ct.uint32), (TM_, 1)), (TM_, TK_)) + mu * TM_
    col = ct.broadcast_to(ct.reshape(ct.arange(TK_, dtype=ct.uint32), (1, TK_)), (TM_, TK_)) + ku * TK_
    h = ((row * 73856093) ^ (col * 19349663) ^ su) * 0x9E3779B1
    h = h ^ (h >> 16); h = h * 0x85EBCA6B; h = h ^ (h >> 13)
    dither = h & 0xFFFF
    u = ct.bitcast(acc, ct.uint32)
    ct.store(dx, index=(m, k), tile=ct.astype((u + dither) >> 16, ct.uint16))


# ── FP8 E4M3 data-gradient (the MAI/DeepSeek dgrad, 2026-06-14) ───────────────
# dx = dy @ Wᵀ in FP8 E4M3 + E8M0 per-32-block scaling (the SAME block scaling as the forward).
# Both dy(M,N) and W(K,N) are quantized per-32 along N (the contraction); the mma transposes the
# W operand AND its scale IN-KERNEL (Option A — ct.mma_scaled accepts transposed fp8+e8m0, probed
# tests/kernels/_probe_fp8_dgrad_transpose.py). E4M3 (not MAI's E5M2) because our fine block scaling
# already handles the dynamic range at the scale layer → the more-precise E4M3 wins ~2× (probed,
# _probe_fp8_dgrad.py). Backward-ONLY → never in the rollout forward → ratio=1 untouched. wgrad
# stays BF16+FP32 (_gemm_dW) — the permanent weight grad keeps higher precision (MAI-matched).

@ct.kernel(occupancy=2)
def _gemm_dx_fp8(dy, dys, W, Ws, dx, NB: ct.Constant[int],
                 TM_: ct.Constant[int], TK_: ct.Constant[int], TN_: ct.Constant[int]):
    """dx = (dy⊙dys) @ (W⊙Ws)ᵀ, MXFP8 in (E4M3 elems + E8M0/32 scales), f32 accum, BF16-bit out.
    dy(M,N) fp8 + dys(M,N//32); W(K,N) fp8 + Ws(K,N//32) (both per-32 along N). Grid (M//TM_,K//TK_),
    reduce over N. W + its scale transposed in-kernel."""
    m, k = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TK_), ct.float32); KS = TN_ // 32
    for n in range(NB):
        a   = ct.bitcast(ct.load(dy,  index=(m, n), shape=(TM_, TN_), latency=10), ct.float8_e4m3fn)
        asc = ct.bitcast(ct.load(dys, index=(m, n), shape=(TM_, KS)), ct.float8_e8m0fnu)
        w   = ct.bitcast(ct.load(W,   index=(k, n), shape=(TK_, TN_), latency=10), ct.float8_e4m3fn)
        ws  = ct.bitcast(ct.load(Ws,  index=(k, n), shape=(TK_, KS)), ct.float8_e8m0fnu)
        acc = ct.mma_scaled(a, asc, ct.transpose(w), ct.transpose(ws), acc)
    ct.store(dx, index=(m, k), tile=ct.bitcast(ct.astype(acc, ct.bfloat16), ct.uint16))


@ct.kernel(occupancy=4)
def _gemm_dW(x, dy, dW, MB: ct.Constant[int],
             TK_: ct.Constant[int], TN_: ct.Constant[int], TM_: ct.Constant[int]):
    """dW = xᵀ @ dy.  Grid (K//TK_, N//TN_), reduce over M.  f32 output.
    ⚠ x is RE-READ by every column block (N//TN_ ×) — at the (V,H) BOUNDARY dW that x is the
    (M,V) gglog/gohot, so TN_=128 (not 64) HALVES the dominant DRAM traffic (ncu: 76% DRAM)."""
    k, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TK_, TN_), ct.float32)
    for m in range(MB):
        tx = ct.bitcast(ct.load(x, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(TM_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ct.transpose(tx), tdy, acc)            # (TK,TM)@(TM,TN)
    ct.store(dW, index=(k, n), tile=acc)


@ct.kernel(occupancy=4)
def _gemm_dW_acc(x, dy, dW, MB: ct.Constant[int],
                 TK_: ct.Constant[int], TN_: ct.Constant[int], TM_: ct.Constant[int]):
    """dW += xᵀ @ dy — _gemm_dW accumulating onto the existing dW (training boundary: the
    input-embed grad lands directly on the LM-head grad, replacing the separate giegr buffer
    (0.62 GB at V=151936) + the _acc_f32 pass; f32 add is commutative-exact ⇒ bitwise == the
    old (giegr + gegrad)). Grid (K//TK_, N//TN_)."""
    k, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TK_, TN_), ct.float32)
    for m in range(MB):
        tx = ct.bitcast(ct.load(x, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tdy = ct.bitcast(ct.load(dy, index=(m, n), shape=(TM_, TN_), latency=10), ct.bfloat16)
        acc = ct.mma(ct.transpose(tx), tdy, acc)
    base = ct.load(dW, index=(k, n), shape=(TK_, TN_))
    ct.store(dW, index=(k, n), tile=base + acc)


# ── tied-LM-head GEMMs (device-resident boundary; embed never leaves the device) ──
# logits = hidden @ embedᵀ  (A(M,K)@B(N,K)ᵀ → C(M,N) f32, for the CE which needs f32 logits).

@ct.kernel(occupancy=2)
def _gemm_nt_f32(A, B, C, KB: ct.Constant[int],
                 TM_: ct.Constant[int], TN_: ct.Constant[int], TK_: ct.Constant[int]):
    """C = A @ Bᵀ, BF16-bit inputs, f32 accumulate + f32 output. A(M,K), B(N,K) → C(M,N).
    Fixed K-order → batch-invariant. (= _gemm but B is read row-major as (N,K) and transposed.)"""
    m, n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM_, TN_), ct.float32)
    for k in range(KB):
        ta = ct.bitcast(ct.load(A, index=(m, k), shape=(TM_, TK_), latency=10), ct.bfloat16)
        tb = ct.bitcast(ct.load(B, index=(n, k), shape=(TN_, TK_), latency=10), ct.bfloat16)
        acc = ct.mma(ta, ct.transpose(tb), acc)             # (TM,TK)@(TK,TN)
    ct.store(C, index=(m, n), tile=acc)


ACM, ACN = 128, 128   # f32 accumulate tile


@ct.kernel
def _acc_f32(src, dst):
    """dst += src (both f32, (R, Cc)). Grid (R//ACM, Cc//ACN). Sums the LM-head + input-embed
    gradient contributions into one (V,H) tied-embedding grad before the AdamW step."""
    m, n = ct.bid(0), ct.bid(1)
    a = ct.load(src, index=(m, n), shape=(ACM, ACN))
    b = ct.load(dst, index=(m, n), shape=(ACM, ACN))
    ct.store(dst, index=(m, n), tile=a + b)


@ct.kernel
def _acc_f32_flat(src, dst):
    """dst += src (f32, (R, Cc)). Grid (R//128, Cc//128) or 1D (N//128, 1) or whatever fits.
    Reads 128 elements at a time to support smaller H·E configurations than 16384."""
    m = ct.bid(0)
    a = ct.load(src, index=(m, 0), shape=(128, 1))
    b = ct.load(dst, index=(m, 0), shape=(128, 1))
    ct.store(dst, index=(m, 0), tile=a + b)


RTM, RTN = 64, 128   # residual-add / cast tile


@ct.kernel
def _residual_add(a, b, out):
    """out = bf16(a + b).  a,b,out BF16 bits (M,H). Grid (M//RTM, H//RTN)."""
    m, n = ct.bid(0), ct.bid(1)
    av = ct.astype(ct.bitcast(ct.load(a, index=(m, n), shape=(RTM, RTN)), ct.bfloat16), ct.float32)
    bv = ct.astype(ct.bitcast(ct.load(b, index=(m, n), shape=(RTM, RTN)), ct.bfloat16), ct.float32)
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(av + bv, ct.bfloat16), ct.uint16))


@ct.kernel
def _residual_add_rf32(a, b, out):
    """FP32-residual add: out(f32) = a(f32 residual) + b(bf16 branch). a,out (M,H) f32; b BF16 bits.
    Keeps the residual stream in fp32 (no coarse bf16 rounding of the ~6912 massive activation across
    layers); the branch output b is the bf16 GEMM result. Grid (M//RTM, H//RTN)."""
    m, n = ct.bid(0), ct.bid(1)
    av = ct.load(a, index=(m, n), shape=(RTM, RTN))
    bv = ct.astype(ct.bitcast(ct.load(b, index=(m, n), shape=(RTM, RTN)), ct.bfloat16), ct.float32)
    ct.store(out, index=(m, n), tile=av + bv)


@ct.kernel
def _cast_bf16(x, out):
    """out = bf16 bits of x (f32), ROUND-to-nearest (ct.astype). Grid (M//RTM, N//RTN)."""
    m, n = ct.bid(0), ct.bid(1)
    v = ct.load(x, index=(m, n), shape=(RTM, RTN))
    ct.store(out, index=(m, n), tile=ct.bitcast(ct.astype(v, ct.bfloat16), ct.uint16))


@ct.kernel
def _trunc_bf16(x, out):
    """out = bf16 bits of x (f32) by TRUNCATION (bits>>16) — bit-identical to ATTENTION.PY's
    f32_to_bf16_bits ONLY. ⚠ NORM.PY's f32_to_bf16_bits is ROUND-TO-NEAREST-EVEN (= _cast_bf16):
    rmsnorm_forward/backward RNE-round their inputs, so a device replacement for the final-norm
    boundary must use _cast_bf16, NOT this. (This mismatch was the decode model's parked
    "removed sync changes the gout value read" mystery — solved 2026-06-11: ~47% of elements
    differ by 1 ulp, biasing Σx²/rstd; never a race.) Grid (M//RTM, N//RTN)."""
    m, n = ct.bid(0), ct.bid(1)
    v = ct.load(x, index=(m, n), shape=(RTM, RTN))
    ct.store(out, index=(m, n), tile=ct.astype(ct.bitcast(v, ct.uint32) >> 16, ct.uint16))


# ── token-major ↔ head-major transpose (for the attention block) ─────────────
# token-major (B*S, H*Dh) ↔ head-major (B*H*S, Dh) — swap S and H axes around attention.
# Key: for fixed (b, head), a contiguous (TT, Dh) tile in token-major (TT consecutive
# positions, that head's Dh columns) maps to a contiguous (TT, Dh) tile in head-major.
# So it's a plain tile-copy with remapped block indices (uint16 bf16 bits). Grid (B*H, NSB).
TT = 64    # transpose tile = positions per block (= BQ)
HD = 128   # head_dim (Qwen3-0.6B real head_dim — hardcoded, single model target)


@ct.kernel
def _tok_to_head(tok, head, Hh: ct.Constant[int], NSB: ct.Constant[int]):
    """tok (B*S, Hh*HD) → head (B*Hh*S, HD).  bid(0)=b*Hh+h, bid(1)=position-block."""
    bh, sb = ct.bid(0), ct.bid(1)
    b = bh // Hh
    ct.store(head, index=(bh * NSB + sb, 0),
             tile=ct.load(tok, index=(b * NSB + sb, bh % Hh), shape=(TT, HD)))


@ct.kernel
def _head_to_tok(head, tok, Hh: ct.Constant[int], NSB: ct.Constant[int]):
    """head (B*Hh*S, HD) → tok (B*S, Hh*HD).  Inverse of _tok_to_head."""
    bh, sb = ct.bid(0), ct.bid(1)
    b = bh // Hh
    ct.store(tok, index=(b * NSB + sb, bh % Hh),
             tile=ct.load(head, index=(bh * NSB + sb, 0), shape=(TT, HD)))


@ct.kernel
def _head_to_tok_f32(head, tok, Hh: ct.Constant[int], NSB: ct.Constant[int]):
    """head (B*Hh*S, HD) f32 → tok (B*S, Hh*HD) BF16 bits. Transpose + cast (attention O→o_proj)."""
    bh, sb = ct.bid(0), ct.bid(1)
    b = bh // Hh
    t = ct.load(head, index=(bh * NSB + sb, 0), shape=(TT, HD))
    ct.store(tok, index=(b * NSB + sb, bh % Hh), tile=ct.bitcast(ct.astype(t, ct.bfloat16), ct.uint16))


DTM = 64   # rows per block for the attention Delta


@ct.kernel
def _cast64(x, out):
    """x (R,HD) f32 → out (R,HD) BF16 bits. Grid (R//DTM,). For attention-bwd f32 dQ/dK → rope-bwd (bf16)."""
    r = ct.bid(0)
    ct.store(out, index=(r, 0), tile=ct.bitcast(ct.astype(ct.load(x, index=(r, 0), shape=(DTM, HD)), ct.bfloat16), ct.uint16))


@ct.kernel
def _add_f32(a, b, o):
    """o = a + b, all (R,HD) f32. Grid (R//DTM,). For the prefix-shared prompt dK/dV
    (self + Σ_G cross) — f32+f32 like the host helper, so the resident path stays bitwise."""
    r = ct.bid(0)
    ct.store(o, index=(r, 0), tile=ct.load(a, index=(r, 0), shape=(DTM, HD)) +
                                   ct.load(b, index=(r, 0), shape=(DTM, HD)))


@ct.kernel
def _embed_gather(ids, embed, out, HB: ct.Constant[int]):
    """out[r] = f32(bf16_embed[ids[r]]) — DEVICE token-id gather (data-dependent row index, the
    probed-OK moe._ggemm pattern). ids (R,1) i32; embed (V, HB*128) bf16 bits; out (R, HB*128) f32.
    Bitwise == the training onehot-GEMM gather (both yield the exact f32 of the bf16 embed row).
    Lets the decode loop feed argmax ids (glab) straight back as the next step's input with NO
    host round-trip — the whole rollout needs ONE sync at the end. Grid (R,)."""
    r = ct.bid(0)
    row = ct.reshape(ct.load(ids, index=(r, 0), shape=(1, 1)), ())
    for hb in range(HB):
        t = ct.astype(ct.bitcast(ct.load(embed, index=(row, hb), shape=(1, 128)), ct.bfloat16), ct.float32)
        ct.store(out, index=(r, hb), tile=t)


@ct.kernel
def _onehot_set(ids, oh):
    """oh[r, ids[r]] = bf16(1.0) bits (0x3F80) — DEVICE onehot build (data-dependent column
    store, the probed scalar-store-index pattern). Caller memsets oh to 0 first. Replaces the
    HOST (M,V) onehot build + upload, which dominated the training forward (~0.3 s/step at
    M=1024: a 1.2 GB numpy scatter + 0.6 GB PCIe). Bits identical to the host path
    (_f32bf(1.0) == 0x3F80) ⇒ the input-embed _gemm_dW stays bitwise. Grid (M,)."""
    r = ct.bid(0)
    c = ct.reshape(ct.load(ids, index=(r, 0), shape=(1, 1)), ())
    ct.store(oh, index=(r, c), tile=ct.full((1, 1), 0x3F80, ct.uint16))


@ct.kernel
def _inc1(x):
    """x[0,0] += 1 (i32). Advances the device position/seed counter at the END of a captured decode
    step so the graph replays at successive positions with no host involvement. Grid (1,)."""
    ct.store(x, index=(0, 0), tile=ct.load(x, index=(0, 0), shape=(1, 1)) + 1)


@ct.kernel
def _bnd_acc(dh, R0: ct.Constant[int], RD: ct.Constant[int], G: ct.Constant[int]):
    """dh[R0] += Σ_{g<G} dh[RD+g]  (f32, fixed order, in place). The prefix-GRPO boundary:
    the G duplicated head rows (copies of hidden row R0 = Sp-1, one per completion — the
    PrefixGrouper include_prefix_last trick) carry per-completion (label, adv); their dhidden
    must flow back into the ONE real row — this is autograd's implicit duplicate-sum made
    explicit. Grid (H // 128,); each block owns a disjoint 128-column slice."""
    n = ct.bid(0)
    acc = ct.load(dh, index=(R0, n), shape=(1, 128))
    for g in range(G):
        acc = acc + ct.load(dh, index=(RD + g, n), shape=(1, 128))
    ct.store(dh, index=(R0, n), tile=acc)


@ct.kernel
def _attn_delta(O, dO, Delta):
    """Delta[i] = Σ_d O[i,d]·dO[i,d] (softmax-Jacobian correction for attention backward).
    O (R,HD) f32, dO (R,HD) BF16 bits → Delta (R,1) f32. Grid (R//DTM,)."""
    r = ct.bid(0)
    o = ct.load(O, index=(r, 0), shape=(DTM, HD))
    do = ct.astype(ct.bitcast(ct.load(dO, index=(r, 0), shape=(DTM, HD)), ct.bfloat16), ct.float32)
    ct.store(Delta, index=(r, 0), tile=ct.sum(o * do, axis=-1, keepdims=True))
