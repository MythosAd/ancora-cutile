"""AdamW cuda-tile kernel — numeric correctness vs numpy over several steps (also
confirms cuda-tile accepts RUNTIME float scalars for the per-step bias corrections)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.optim.adamw import AdamW

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def test():
    print("--- AdamW kernel vs numpy (fp32 master, runtime bias-correction) ---")
    rng = np.random.default_rng(0)
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.01
    # two params: a matrix-ish (256,64)→numel 16384 and a tiny vector (64,)
    p_w = (rng.standard_normal((256, 64)) * 0.1).astype(np.float32)
    p_n = (1.0 + rng.standard_normal(64) * 0.05).astype(np.float32)
    opt = AdamW({"w": p_w.copy(), "n": p_n.copy()}, lr=lr, betas=(b1, b2), eps=eps,
                wd=wd, no_decay=("n",))

    ref = {"w": [p_w.copy(), 0.0, 0.0, wd], "n": [p_n.copy(), 0.0, 0.0, 0.0]}
    ok = True
    for t in range(1, 6):
        grads = {"w": (rng.standard_normal((256, 64)) * 0.5).astype(np.float32),
                 "n": (rng.standard_normal(64) * 0.5).astype(np.float32)}
        opt.step(grads, si)
        master = opt.master()
        for name in ("w", "n"):
            p, m, v, wdi = ref[name]; g = grads[name]
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            p = p - lr * (mh / (np.sqrt(vh) + eps) + wdi * p)
            ref[name] = [p, m, v, wdi]
            e = np.abs(master[name] - p).max() / (np.abs(p).max() + 1e-9)
            o = e < 1e-5; ok &= o
            if t in (1, 5):
                print(f"  step {t} {name}: rel {e:.2e}  {'OK' if o else 'FAIL'}")
    # bf16 weights == bf16(master)
    w16 = opt.weights()["w"]
    from ancora.optim.adamw import _bf16_bits, _bits_f32
    e = np.abs(w16 - _bits_f32(_bf16_bits(opt.master()["w"]))).max()
    print(f"  weights() = bf16(master): max diff {e:.2e}  {'OK' if e == 0 else 'FAIL'}")
    ok &= (e == 0)
    opt.free()
    return ok


if __name__ == "__main__":
    print("AdamW (cuda-tile)")
    print("=" * 50)
    ok = test()
    print("=" * 50)
    print(f"  {'PASS' if ok else 'FAIL'}")
