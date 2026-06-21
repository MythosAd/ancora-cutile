"""
ancora/kernels/quant_nvfp4.py — host NVFP4 quantization for the CUTLASS rollout GEMM path.

NVFP4 = FP4 (E2M1) data, packed 2/byte, with a per-16-element E4M3 (ue4m3) scale VALUE
(not an exponent — distinct from MXFP8's E8M0). More accurate than MXFP4 (E8M0 block-32):
the E4M3 mantissa'd scale lands on the block amax instead of rounding up to a power of 2,
and block-16 limits outlier damage. See [[precision-format-decision]].

E2M1 magnitudes (by exp<<1|mant): {0,.5,1,1.5,2,3,4,6}, max 6.0.
The E4M3 scale bytes are produced by quant.f32_to_e4m3 — bit-identical to CUTLASS
float_ue4m3_t for positive values (both use the cvt.e4m3x2 encoding). Verified in
tests/hardware/test_nvfp4_quant.py (dequant ≈ input, ~9.5% RMS = inherent FP4 error).
"""
import numpy as np
from ancora.kernels.quant import f32_to_e4m3, e4m3_to_f32

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float64)   # by (exp<<1|mant)
E2M1_MAX = 6.0
NB = 16   # NVFP4 block size


def f32_to_e2m1(x):
    """float32 → 4-bit E2M1 codes (uint8 0..15), round-to-nearest of the 8 magnitudes."""
    x = np.asarray(x, np.float64); ax = np.abs(x)
    idx = np.clip(np.searchsorted(E2M1_MAG, ax), 1, 7)
    lo, hi = idx - 1, idx
    pick_hi = (ax - E2M1_MAG[lo]) > (E2M1_MAG[hi] - ax)
    mag = np.where(pick_hi, hi, lo).astype(np.uint8)
    return mag | ((np.signbit(x).astype(np.uint8)) << 3)


def e2m1_to_f32(code):
    code = np.asarray(code, np.uint8)
    mag = E2M1_MAG[code & 0x7]
    return np.where((code & 0x8) != 0, -mag, mag).astype(np.float32)


def pack_fp4(codes):
    """(.., even) 4-bit codes → packed bytes (2 per byte, low nibble = first element)."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2).astype(np.uint8)
    return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)


def unpack_fp4(packed):
    lo = packed & 0xF; hi = (packed >> 4) & 0xF
    return np.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_nvfp4_rowblock(x):
    """x (M,K) f32 → packed fp4 (M,K//2) u8 + E4M3 scale (M,K//16) u8. Scale = amax/6 → E4M3."""
    M, K = x.shape
    xb = x.reshape(M, K // NB, NB)
    amax = np.abs(xb).max(-1, keepdims=True)
    sc_e4m3 = f32_to_e4m3(np.maximum(amax / E2M1_MAX, 1e-12)[..., 0])     # (M,K//16) E4M3 bytes
    sc_val = e4m3_to_f32(sc_e4m3)[..., None]                              # actual stored scale value
    codes = f32_to_e2m1(xb / np.maximum(sc_val, 1e-12)).reshape(M, K)
    return pack_fp4(codes), sc_e4m3.astype(np.uint8)


def dequantize_nvfp4(packed, sc_e4m3):
    """For reference checks: reconstruct (M,K) f32 from packed fp4 + E4M3 scale."""
    M = packed.shape[0]; K = packed.shape[1] * 2
    fp4 = e2m1_to_f32(unpack_fp4(packed)).reshape(M, K // NB, NB)
    return (fp4 * e4m3_to_f32(sc_e4m3)[:, :, None]).reshape(M, K)
