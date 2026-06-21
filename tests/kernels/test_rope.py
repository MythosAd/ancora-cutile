"""RoPE forward/backward correctness (vs numpy rotate-half on BF16-rounded inputs) +
batch invariance (a token's rotation is identical regardless of batch/seq context).

Keep this around: re-run after every cuda-tile / toolkit upgrade."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.rope import (rope_forward, rope_backward, build_cos_sin,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32, RTM)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def _bf(x): return bf32(f32bf(x))


def rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def ref_fwd(x, base):
    """HF Qwen3/Llama apply_rotary_pos_emb on BF16-rounded x (cos/sin width-D, duplicated)."""
    Bb, H, S, D = x.shape
    c, s = build_cos_sin(S, D, base)             # (S, D/2)
    cosf = np.concatenate([c, c], -1)[None, None]  # (1,1,S,D)
    sinf = np.concatenate([s, s], -1)[None, None]
    xb = _bf(x).astype(np.float64)
    return xb * cosf + rotate_half(xb) * sinf


def ref_bwd(dy, base):
    """dx = dy*cos + rotate_half_T(dy)*sin; rotate_half^T(v)=cat(v[d:], -v[:d])."""
    Bb, H, S, D = dy.shape; d = D // 2
    c, s = build_cos_sin(S, D, base)
    cosf = np.concatenate([c, c], -1)[None, None]
    sinf = np.concatenate([s, s], -1)[None, None]
    dyb = _bf(dy).astype(np.float64)
    rot_t = np.concatenate([dyb[..., d:], -dyb[..., :d]], -1)
    return dyb * cosf + rot_t * sinf


def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def test_correctness():
    print("--- correctness (vs numpy rotate-half, BF16-rounded) ---")
    rng = np.random.default_rng(0); ok = True; base = 1e6
    for (Bb, H, S, D) in [(1, 16, 128, 128), (2, 8, 256, 128), (1, 16, 2048, 64)]:
        x  = (rng.standard_normal((Bb, H, S, D)) * 1.0).astype(np.float32)
        dy = (rng.standard_normal((Bb, H, S, D)) * 1.0).astype(np.float32)
        y  = rope_forward(x, si, base)
        dx = rope_backward(dy, si, base)
        ry, rdx = rel(_bf(y), ref_fwd(x, base)), rel(_bf(dx), ref_bwd(dy, base))
        o = ry < 0.02 and rdx < 0.02; ok &= o
        print(f"  B={Bb} H={H:2d} S={S:5d} D={D:3d}: fwd={ry*100:.2f}% "
              f"bwd={rdx*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_batch_invariance():
    """Token at (head 0, pos 0..S0) must rotate identically whether seq len is S0 or 4·S0
    (RoPE is elementwise per position — no cross-token coupling)."""
    print("--- batch/seq invariance (token bits identical) ---")
    rng = np.random.default_rng(2); H = 16; D = 128; base = 1e6; S0 = 128

    def run(S):
        x = np.zeros((1, H, S, D), np.float32)
        head0 = np.random.default_rng(7).standard_normal((S0, D)).astype(np.float32)
        x[0, 0, :S0] = head0                                   # fixed first S0 positions
        x[0, 0, S0:] = rng.standard_normal((S - S0, D)).astype(np.float32)
        y = rope_forward(x, si, base)
        return f32bf(y[0, 0, :S0])

    a = run(S0)
    b = run(4 * S0)
    same = np.array_equal(a, b)
    print(f"  S={S0} vs S={4*S0}: token bits identical={same}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    print(f"RoPE — RTM={RTM} (rotate-half / NEOX, Qwen3 rope_theta=1e6)")
    print("=" * 60)
    ok = test_correctness()
    bi = test_batch_invariance()
    print("=" * 60)
    print(f"  correctness {'PASS' if ok else 'FAIL'} | batch-invariance {'PASS' if bi else 'FAIL'}")
