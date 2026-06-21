"""
ancora/kernels/quant.py — MXFP8 quantization (host) for the forward GEMM path.

MXFP8 (OCP microscaling): per block of 32 elements, one E8M0 (power-of-2) shared scale;
each element in E4M3 (float8_e4m3fn, max 448). The cuda-tile `_fwd_mxfp8` kernel reads
the FP8 bytes (bitcast→float8_e4m3fn) and the E8M0 scale bytes and does the block-scaled
MMA. This module produces those bytes from float32.

Decision (2026-05 research): MXFP8 for the training+rollout forward — FP4 training isn't a
stable path yet and cuda-tile 1.4.0 can't load packed FP4 (issue #47); future 4-bit target
is NVFP4, not MXFP4. See [[precision-format-decision]].

No ml_dtypes available → E4M3 encoded via an exact 256-entry lookup (round-to-nearest).
"""
import numpy as np
import cuda.tile as ct

B = 32          # MXFP8 scale block
E4M3_MAX = 448.0
QTM = 64        # rows per block for the on-GPU activation quant kernel
QWN = 128       # columns per block for the on-GPU weight quant kernel


# ── on-GPU MXFP8 activation quantization ─────────────────────────────────────
# Per 32-elem block: E8M0 scale byte = (f32 biased exponent of amax) - 7  (smallest
# power-of-2 scale that keeps amax/scale ≤ 256 ≤ 448; FP precision is relative so the
# exact 448-fill doesn't matter). Then astype(x/scale, float8_e4m3fn). cuda-tile has no
# floor/ceil, so we read the exponent bits directly (probed: bitcast→uint32, >>, & work).

@ct.kernel
def _quant_mxfp8(x, fp8_out, scale_out, KB: ct.Constant[int]):
    """x:(M,K) BF16 bits → fp8_out:(M,K) u8 (E4M3) + scale_out:(M,K//32) u8 (E8M0 byte).
    Grid (M // QTM,)."""
    m = ct.bid(0)
    for kb in range(KB):
        xt = ct.astype(ct.bitcast(ct.load(x, index=(m, kb), shape=(QTM, B)), ct.bfloat16), ct.float32)
        amax = ct.max(ct.maximum(xt, 0.0 - xt), axis=-1, keepdims=True)            # (QTM,1)
        ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF                            # biased f32 exp
        byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((QTM, 1), 0, ct.uint32))
        sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
        fp8 = ct.bitcast(ct.astype(xt / ct.broadcast_to(sc, (QTM, B)), ct.float8_e4m3fn), ct.uint8)
        ct.store(fp8_out, index=(m, kb), tile=fp8)
        ct.store(scale_out, index=(m, kb), tile=ct.astype(byte, ct.uint8))


# ── on-GPU MXFP8 WEIGHT quantization (col-block: 32 along K=axis 0) ───────────
# Mirrors _quant_mxfp8 but the 32-block runs along K (axis 0), giving the (K,N) fp8 + (K//32,N) E8M0
# layout `_fwd_mxfp8`'s w/w_scale expects (weight pre-transposed (K,N)). Same FLOOR-exponent E8M0 as
# _quant_mxfp8, so a device re-quant during MXFP8 training and the rollout's weight quant produce
# IDENTICAL bytes from the same bf16 weight → rollout==training stays bitwise. (Differs from the host
# quantize_colblock's ceil convention, but each is self-consistent; mma_scaled decodes whatever byte.)
# Lets the MXFP8 training loop re-quantize the AdamW-updated weights ON DEVICE — no host round-trip,
# no alloc churn — so the forward GEMMs actually track the weight updates.

@ct.kernel
def _quant_mxfp8_w(w, fp8_out, scale_out):
    """w (K,N) BF16 bits → fp8_out (K,N) u8 (E4M3) + scale_out (K//32,N) u8 (E8M0). Block of 32 along
    K (axis 0). Grid (K//32, N//QWN)."""
    kb, nb = ct.bid(0), ct.bid(1)
    wt = ct.astype(ct.bitcast(ct.load(w, index=(kb, nb), shape=(B, QWN)), ct.bfloat16), ct.float32)
    amax = ct.max(ct.maximum(wt, 0.0 - wt), axis=0, keepdims=True)              # (1, QWN) over the 32 K-rows
    ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
    byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((1, QWN), 0, ct.uint32))
    sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
    fp8 = ct.bitcast(ct.astype(wt / ct.broadcast_to(sc, (B, QWN)), ct.float8_e4m3fn), ct.uint8)
    ct.store(fp8_out, index=(kb, nb), tile=fp8)
    ct.store(scale_out, index=(kb, nb), tile=ct.astype(byte, ct.uint8))


# ── on-GPU MXFP8 WEIGHT quant, COLUMN-MAJOR output (for CUTLASS B = (N,K) row-major) ──────────
# Identical reduction to _quant_mxfp8_w (32-block along K=axis 0, same FLOOR-exponent E8M0) but the
# output is TRANSPOSED: fp8_out (N,K) + scale_out (N,K//32). CUTLASS's blockscaled GEMM takes B as
# column-major (K,N) == row-major (N,K) with scale (N,K//32). Same bf16 master + same kernel in both
# the training-prefill and rollout-decode layers ⇒ byte-identical weight ⇒ ratio=1 holds (the GEMM
# itself is M-invariant, tests/hardware/test_cutlass_minvariant.py).

@ct.kernel
def _quant_mxfp8_w_cm(w, fp8_out, scale_out):
    """w (K,N) BF16 bits → fp8_out (N,K) u8 (E4M3) + scale_out (N,K//32) u8 (E8M0). Block of 32 along
    K (axis 0); output transposed (N-major). Grid (K//32, N//QWN)."""
    kb, nb = ct.bid(0), ct.bid(1)
    wt = ct.astype(ct.bitcast(ct.load(w, index=(kb, nb), shape=(B, QWN)), ct.bfloat16), ct.float32)
    amax = ct.max(ct.maximum(wt, 0.0 - wt), axis=0, keepdims=True)              # (1, QWN) over the 32 K-rows
    ea = (ct.bitcast(amax, ct.uint32) >> 23) & 0xFF
    byte = ct.where(ct.greater_equal(ea, 7), ea - 7, ct.full((1, QWN), 0, ct.uint32))
    sc = ct.exp2(ct.astype(byte, ct.float32) - 127.0)
    fp8 = ct.astype(wt / ct.broadcast_to(sc, (B, QWN)), ct.float8_e4m3fn)       # (B, QWN) fp8
    ct.store(fp8_out, index=(nb, kb), tile=ct.bitcast(ct.transpose(fp8), ct.uint8))            # (QWN, B) → (N,K)
    ct.store(scale_out, index=(nb, kb), tile=ct.transpose(ct.astype(byte, ct.uint8)))          # (QWN, 1) → (N,K//32)


def _e4m3_table():
    """The 256 E4M3fn values (NaN at 0x7F/0xFF). Byte order is magnitude-monotonic."""
    v = np.zeros(256, np.float64)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            val = (m / 8.0) * 2.0 ** (-6)              # subnormal
        elif e == 15 and m == 7:
            val = np.nan                                # NaN
        else:
            val = (1.0 + m / 8.0) * 2.0 ** (e - 7)      # normal
        v[b] = s * val
    return v.astype(np.float32)


_TAB = _e4m3_table()
_POS = _TAB[0:127].astype(np.float64)   # bytes 0x00..0x7E → 0 .. 448 ascending


def f32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """float32 → E4M3fn bytes (uint8), round-to-nearest via the magnitude lookup."""
    x = np.asarray(x, np.float32)
    ax = np.minimum(np.abs(x).astype(np.float64), E4M3_MAX)
    idx = np.clip(np.searchsorted(_POS, ax), 1, 126)
    lo, hi = idx - 1, idx
    pick_hi = (ax - _POS[lo]) > (_POS[hi] - ax)
    byte = np.where(pick_hi, hi, lo).astype(np.uint8)
    sign = (np.signbit(x).astype(np.uint8)) << 7
    return (byte | sign).astype(np.uint8)


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    return _TAB[np.asarray(b, np.uint8)]


def _e8m0_scale(amax):
    """Choose a power-of-2 scale so amax/scale ≤ 448. Returns (scale_byte u8, scale f32).
    E8M0: value = 2^(byte-127); byte = ceil(log2(amax/448)) + 127."""
    amax = np.maximum(amax.astype(np.float32), 1e-30)
    e = np.ceil(np.log2(amax / E4M3_MAX))
    sb = np.clip(e + 127, 0, 255).astype(np.uint8)
    return sb, (2.0 ** (sb.astype(np.float32) - 127))


def quantize_rowblock(x: np.ndarray):
    """Activations x:(M,K) → (fp8 (M,K) u8, scale (M,K//32) u8). Blocks of 32 along K."""
    M, K = x.shape
    xb = x.reshape(M, K // B, B)
    sb, sc = _e8m0_scale(np.abs(xb).max(-1, keepdims=True))     # (M,K//32,1)
    fp8 = f32_to_e4m3(xb / sc).reshape(M, K)
    return fp8, sb.reshape(M, K // B)


def quantize_colblock(w: np.ndarray):
    """Weights w:(K,N) → (fp8 (K,N) u8, scale (K//32,N) u8). Blocks of 32 along K (axis 0).
    Matches `_fwd_mxfp8`'s w(K,N), w_scale(K//32,N) convention (weight pre-transposed)."""
    K, N = w.shape
    wb = w.reshape(K // B, B, N)
    sb, sc = _e8m0_scale(np.abs(wb).max(1, keepdims=True))       # (K//32,1,N)
    fp8 = f32_to_e4m3(wb / sc).reshape(K, N)
    return fp8, sb.reshape(K // B, N)


def dequantize_rowblock(fp8, scale):
    """For reference checks: reconstruct (M,K) f32 from MXFP8 bytes."""
    M, K = fp8.shape
    return (e4m3_to_f32(fp8).reshape(M, K // B, B) *
            (2.0 ** (scale.astype(np.float32) - 127))[:, :, None]).reshape(M, K)
