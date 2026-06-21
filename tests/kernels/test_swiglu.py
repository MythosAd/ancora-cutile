"""SwiGLU forward + backward vs numpy fp64 (BF16-rounded inputs). Keep — re-run after
toolkit changes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.activation import (swiglu_forward, swiglu_backward,
                                        f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32)

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def _bf(x): return bf32(f32bf(x))
def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def test():
    print("--- SwiGLU fwd/bwd vs numpy fp64 ---")
    rng = np.random.default_rng(0); ok = True
    for (M, I) in [(128, 256), (256, 3072)]:
        g = (rng.standard_normal((M, I)) * 1.0).astype(np.float32)
        u = (rng.standard_normal((M, I)) * 1.0).astype(np.float32)
        do = (rng.standard_normal((M, I)) * 0.5).astype(np.float32)
        gb, ub, dob = _bf(g).astype(np.float64), _bf(u).astype(np.float64), _bf(do).astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-gb)); silu = gb * sig
        y_r = silu * ub
        dg_r = dob * ub * (sig * (1.0 + gb * (1.0 - sig)))
        du_r = dob * silu

        y = swiglu_forward(g, u, si)
        dg, du = swiglu_backward(g, u, do, si)
        ry, rdg, rdu = rel(_bf(y), y_r), rel(_bf(dg), dg_r), rel(_bf(du), du_r)
        o = ry < 0.03 and rdg < 0.03 and rdu < 0.03; ok &= o
        print(f"  M={M} I={I}: fwd={ry*100:.2f}% dgate={rdg*100:.2f}% dup={rdu*100:.2f}%  "
              f"{'OK' if o else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("SwiGLU (silu(gate)*up)")
    print("=" * 50)
    ok = test()
    print("=" * 50)
    print(f"  {'PASS' if ok else 'FAIL'}")
