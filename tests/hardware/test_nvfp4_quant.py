"""STEP 1 of the NVFP4 rollout GEMM path: host NVFP4 quantization, verified in isolation.
NVFP4 = FP4 (E2M1) data + E4M3 per-16-block scale VALUE (not an exponent). E2M1 magnitudes:
{0,.5,1,1.5,2,3,4,6} (max 6). Verify dequant(fp4, e4m3 scale) ≈ input within FP4 error."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
from ancora.kernels.quant import f32_to_e4m3, e4m3_to_f32   # reuse our verified E4M3 codec

# ── FP4 E2M1 codec ──
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float64)   # by (exp<<1|mant)
E2M1_MAX = 6.0
NB = 16   # NVFP4 block size

def f32_to_e2m1(x):
    """float32 → 4-bit E2M1 codes (uint8 0..15), round-to-nearest-even-ish via midpoints."""
    x = np.asarray(x, np.float64); ax = np.abs(x)
    # nearest magnitude index (round to nearest of the 8 levels)
    idx = np.searchsorted(E2M1_MAG, ax)               # insertion point in [0,8]
    idx = np.clip(idx, 1, 7)
    lo, hi = idx - 1, idx
    pick_hi = (ax - E2M1_MAG[lo]) > (E2M1_MAG[hi] - ax)
    mag = np.where(pick_hi, hi, lo).astype(np.uint8)
    sign = (np.signbit(x).astype(np.uint8)) << 3
    return mag | sign                                  # 4-bit code

def e2m1_to_f32(code):
    code = np.asarray(code, np.uint8)
    mag = E2M1_MAG[code & 0x7]
    return np.where((code & 0x8) != 0, -mag, mag).astype(np.float32)

def pack_fp4(codes):
    """(.., even) 4-bit codes → packed bytes (2 per byte, low nibble first)."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2).astype(np.uint8)
    return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)

def unpack_fp4(packed):
    lo = packed & 0xF; hi = (packed >> 4) & 0xF
    return np.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_nvfp4_rowblock(x):
    """x (M,K) → packed fp4 (M,K//2) u8 + E4M3 scale (M,K//16) u8. Scale = amax/6 quantized to E4M3."""
    M, K = x.shape
    xb = x.reshape(M, K // NB, NB)
    amax = np.abs(xb).max(-1, keepdims=True)                      # (M,K//16,1)
    sc_f = np.maximum(amax / E2M1_MAX, 1e-12)                     # ideal scale
    sc_e4m3 = f32_to_e4m3(sc_f[..., 0])                           # (M,K//16) E4M3 bytes
    sc_val = e4m3_to_f32(sc_e4m3)[..., None]                      # the ACTUAL stored scale value
    codes = f32_to_e2m1(xb / np.maximum(sc_val, 1e-12)).reshape(M, K)
    return pack_fp4(codes), sc_e4m3.astype(np.uint8)

def dequantize_nvfp4(packed, sc_e4m3):
    M = packed.shape[0]; K = packed.shape[1] * 2
    fp4 = e2m1_to_f32(unpack_fp4(packed)).reshape(M, K // NB, NB)
    sc = e4m3_to_f32(sc_e4m3)[:, :, None]
    return (fp4 * sc).reshape(M, K)


if __name__ == "__main__":
    print("NVFP4 host quantization — step 1 verification"); print("=" * 60)
    rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)
    rng = np.random.default_rng(0); ok = True
    # codec round-trip on the exact grid
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -3, -6], np.float32)
    assert np.allclose(e2m1_to_f32(f32_to_e2m1(grid)), grid), "E2M1 codec exact-grid FAIL"
    print("  E2M1 codec exact on the 8-level grid: OK")
    for (M, K) in [(256, 512), (512, 1024), (8192, 1024)]:
        x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
        p, s = quantize_nvfp4_rowblock(x)
        deq = dequantize_nvfp4(p, s)
        e = rms(deq, x); o = e < 0.12   # FP4 RMS error is inherently ~6-10%
        ok &= o
        print(f"  M={M} K={K}: dequant vs input rms {e*100:.2f}%  packed {p.shape} scale {s.shape}  {'OK' if o else 'FAIL'}")
    print("=" * 60); print(f"  {'PASS' if ok else 'FAIL'}")
