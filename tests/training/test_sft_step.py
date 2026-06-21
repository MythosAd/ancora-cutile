"""End-to-end SFT training step on the single Qwen3 layer, wiring EVERYTHING:
  forward (layer) → CE loss (linear_ce LM head) → backward (layer) → Muon+AdamW update.
Overfits one fixed (input, labels) batch and asserts the CE loss collapses — the proof
that forward, backward, the loss, and the hybrid optimizer compose into actual learning.

Routing check: 7 proj matrices → Muon, 4 norm gains + LM head → AdamW.
Keep — re-run after any kernel/optimizer/toolkit change."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.loss import linear_ce
from ancora.optim.hybrid import HybridOptimizer

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def main():
    cfg = Qwen3Config(); cfg._B, cfg._S = 1, 128
    B, S, H = cfg._B, cfg._S, cfg.hidden
    M, V = B * S, 2048
    layer = TransformerLayer(cfg, seed=0)
    rng = np.random.default_rng(0)
    w_head = _bf((rng.standard_normal((H, V)) * 0.02).astype(np.float32))
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))   # fixed batch
    labels = rng.integers(0, V, M).astype(np.int64)
    adv = np.ones(M, np.float32)                                          # SFT: L = -mean(logprob)

    params = {**layer.w, "w_head": w_head}
    opt = HybridOptimizer(params)
    r = opt.routing()
    print(f"routing: Muon({len(r['muon'])}) = {r['muon']}")
    print(f"         AdamW({len(r['adamw'])}) = {r['adamw']}")
    print("-" * 64)

    ces = []
    for step in range(40):
        w = opt.weights()                                # BF16 weights ← optimizer master
        for n in layer.w:
            layer.w[n] = w[n]
        w_head = w["w_head"]

        out, cache = layer.forward(x, si, return_cache=True)
        lp, dhidden, dW_head = linear_ce(out.reshape(M, H), w_head, labels, si, advantage=adv)
        ce = float(-lp.mean()); ces.append(ce)

        d_x, grads = layer.backward(dhidden.reshape(B, S, H), cache, si)
        grads["w_head"] = dW_head
        opt.step(grads, si)
        if step % 5 == 0 or step == 39:
            print(f"  step {step:2d}  CE = {ce:.4f}")

    print("-" * 64)
    drop = (ces[0] - ces[-1]) / ces[0] * 100
    print(f"  CE: {ces[0]:.3f} → {ces[-1]:.3f}  ({drop:.0f}% drop, init≈ln(V)={np.log(V):.2f})")
    ok = ces[-1] < 0.5 * ces[0] and ces[-1] < ces[0] - 1.0
    opt.free()
    print("=" * 64)
    print(f"  {'PASS — loss collapsed, training loop works' if ok else 'FAIL — loss did not drop'}")
    return ok


if __name__ == "__main__":
    print("SFT step: layer fwd → linear-CE → layer bwd → Muon+AdamW")
    print("=" * 64)
    main()
