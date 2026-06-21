"""END-TO-END SFT on the REAL Qwen3-0.6B weights (C:\\model\\Qwen3-0.6B), head_dim=128, V=151936.
This is the payoff of the D=128 attention work: the framework's full fwd→loss→bwd→Muon/AdamW
stack run on the ACTUAL pretrained weights (not random init), proving the kernels are correct at
the real model's shapes (hidden=1024, 16 Q / 8 KV heads, head_dim=128, intermediate=3072).

Validation = overfit a fixed (ids, labels) batch: CE must collapse from ≈ln(V)=11.93 toward 0.
That a real-shaped, real-valued network drives its own loss to zero exercises every kernel
(embed gather/scatter, RMSNorm, QK-Norm, RoPE, D=128 flash-attn fwd+bwd, SwiGLU, GEMMs, linear-CE)
on the real weights. (Random token ids — no tokenizer pkg — so the *initial* CE is just ln(V);
the collapse is the signal. A real prompt would only lower the starting CE.)

Run:  python tests/training/test_real_qwen3_sft.py [n_layers] [n_steps]
"""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.qwen3_model import Qwen3Model
from ancora.model.load_qwen3 import load_qwen3
from ancora.optim.hybrid import HybridOptimizer

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def to_flat(w, nl):
    """load_qwen3 dict → Qwen3Model.load() flat format. Names already match the layer keys."""
    flat = {"embed": w["embed"], "lm_head": w["lm_head"], "final_norm": w["final_norm"]}
    for i in range(nl):
        for n, val in w["layers"][i].items():
            flat[f"layer{i}.{n}"] = val
    return flat


def main(NL=6, STEPS=15):
    cfg = Qwen3Config()      # head_dim=128, hidden=1024, 16/8 heads, intermediate=3072, eps=1e-6, theta=1e6
    V = 151936
    B, S, H, M = 1, 128, cfg.hidden, 128
    assert cfg.head_dim == 128 and cfg.hidden == 1024

    print(f"loading REAL Qwen3-0.6B  ({NL} of 28 layers, V={V}, head_dim={cfg.head_dim})...")
    t0 = time.time()
    raw = load_qwen3(n_layers=NL)            # f32 (already BF16-valued bits)
    flat = to_flat(raw, NL)
    print(f"  loaded in {time.time()-t0:.0f}s")

    model = Qwen3Model(cfg, n_layers=NL, vocab=V, seed=0)   # random init (overwritten by load)
    model.load(flat)
    del raw, flat; gc.collect()              # free the loader copies before the optimizer copies

    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, V, (B, S)).astype(np.int64)
    labels = rng.integers(0, V, M).astype(np.int64)

    opt = HybridOptimizer(model.params())
    r = opt.routing()
    print(f"routing: Muon {len(r['muon'])} (proj matrices) | AdamW {len(r['adamw'])} (embed/head/norms)")
    print("-" * 64)

    ces = []
    for step in range(STEPS):
        ts = time.time()
        model.load(opt.weights())
        hidden, cache = model.forward(input_ids, si)
        ce, grads = model.loss_backward(hidden, labels, cache, si)
        opt.step(grads, si)
        ces.append(ce)
        print(f"  step {step:2d}  CE = {ce:7.4f}   ({time.time()-ts:.1f}s)")

    print("-" * 64)
    drop = (ces[0] - ces[-1]) / ces[0] * 100
    print(f"  CE: {ces[0]:.3f} → {ces[-1]:.3f}  ({drop:.0f}% drop, init≈ln(V)={np.log(V):.2f})")
    ok = ces[-1] < 0.5 * ces[0] and ces[-1] < ces[0] - 1.0
    opt.free()
    print("=" * 64)
    print(f"  {'PASS — REAL Qwen3-0.6B trains end-to-end at D=128' if ok else 'FAIL — loss did not drop'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    print("REAL Qwen3-0.6B SFT overfit — embed → N real layers → norm → LM head → CE")
    print("=" * 64)
    main(NL, STEPS)
