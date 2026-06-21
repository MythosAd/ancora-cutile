"""Full Qwen3 model (embed + N layers + final norm + LM head) SFT overfit — proves the
whole stack's forward + backward + Muon/AdamW compose into learning (loss collapses).
Validates: embedding gather/scatter, multi-layer gradient flow, final norm, LM head.
Uses a small model (N layers, small vocab) for speed; real = 28 layers, V=151936."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.qwen3_model import Qwen3Model
from ancora.optim.hybrid import HybridOptimizer

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def main():
    cfg = Qwen3Config(); cfg._B, cfg._S = 1, 128
    B, S, H, V, NL = 1, 128, cfg.hidden, 2048, 3
    M = B * S
    model = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0)
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, V, (B, S)).astype(np.int64)
    labels = rng.integers(0, V, M).astype(np.int64)

    opt = HybridOptimizer(model.params())
    r = opt.routing()
    print(f"model: {NL} layers, H={H}, V={V}  |  params: {len(model.params())}")
    print(f"routing: Muon {len(r['muon'])} (proj matrices) | AdamW {len(r['adamw'])} (embed/head/norms)")
    print("-" * 64)

    ces = []
    for step in range(40):
        model.load(opt.weights())
        hidden, cache = model.forward(input_ids, si)
        ce, grads = model.loss_backward(hidden, labels, cache, si)
        opt.step(grads, si)
        ces.append(ce)
        if step % 5 == 0 or step == 39:
            print(f"  step {step:2d}  CE = {ce:.4f}")

    print("-" * 64)
    drop = (ces[0] - ces[-1]) / ces[0] * 100
    print(f"  CE: {ces[0]:.3f} → {ces[-1]:.3f}  ({drop:.0f}% drop, init≈ln(V)={np.log(V):.2f})")
    ok = ces[-1] < 0.5 * ces[0] and ces[-1] < ces[0] - 1.0
    opt.free()
    print("=" * 64)
    print(f"  {'PASS — full model trains end-to-end' if ok else 'FAIL — loss did not drop'}")
    return ok


if __name__ == "__main__":
    print("Full Qwen3 model SFT overfit: embed → N layers → norm → LM head → CE")
    print("=" * 64)
    main()
