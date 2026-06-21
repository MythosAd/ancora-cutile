"""long_context (activation checkpointing) on the GRPO prefix path (task #16d / B). The backward
RECOMPUTES each prefix layer's forward from a stored gx_in instead of keeping all NL layers'
intermediates resident — recompute is deterministic ⇒ grads must be BITWISE-identical to full-store
(the correctness gate, exactly as test_checkpoint.py proves for the SFT model). This is what lets RL
TRAINING reach 16K (the SFT path's wall) instead of OOMing at full-store.

Gate: prefix GRPO with long_context=True vs False → scored logprobs Δ=0, layer/MoE/embed grads Δ=0.
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import from_host
from ancora.model.resident_prefix_model import ResidentPrefixMoEModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)


def run(cfg, w, prompt, comps, adv_g, Sp, Sc, G, long_context):
    pre = ResidentPrefixMoEModel(cfg, copy.deepcopy(w), Sp, Sc, G, long_context=long_context)
    h = pre.forward_prefix(prompt, comps, si)
    ce, lp = pre.grpo_loss_backward(h, comps, adv_g, si); sync()
    grads = {"lp": lp.copy(), "embed": pre.gegrad.to_numpy().copy()}
    for j in range(cfg.n_layers):
        grads[f"q{j}"] = pre.layers[j].G["q_proj"].to_numpy().copy()
        if hasattr(pre.layers[j], "moe"):
            grads[f"dWd{j}"] = pre.layers[j].moe.dWd.to_numpy().copy()
    return ce, grads


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)   # (L,D)(G,M)(L,D)(G,M)
    Sp, Sc, G = 128, 64, 4; S = Sp + Sc
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    print("schedule (is_global, ffn_dense):", host.sched)
    w = from_host(host, 1, S)
    rng = np.random.default_rng(3)
    prompt = rng.integers(0, cfg.vocab, size=(Sp,)).astype(np.int64)
    comps = rng.integers(0, cfg.vocab, size=(G, Sc)).astype(np.int64)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    adv_g = ((r - r.mean()) / (r.std() + 1e-6)).astype(np.float32)

    ce_f, g_f = run(cfg, w, prompt, comps, adv_g, Sp, Sc, G, False)   # full-store reference
    ce_c, g_c = run(cfg, w, prompt, comps, adv_g, Sp, Sc, G, True)    # checkpoint (recompute)

    print("-" * 64)
    ok = True
    for k in g_f:
        d = float(np.abs(g_f[k].astype(np.float64) - g_c[k].astype(np.float64)).max())
        bad = d != 0.0
        ok &= not bad
        if bad or k in ("lp", "embed", "q0", "dWd1"):
            print(f"  {k:7s}: max|Δ| = {d:.3e}  {'BITWISE' if not bad else 'DIFFER'}")
    print(f"  CE: full {ce_f:.5f}  ckpt {ce_c:.5f}  Δ={abs(ce_f-ce_c):.1e}")
    print("=" * 64)
    print(f"  {'PASS — prefix long_context BITWISE == full-store (recompute deterministic ⇒ ratio=1 + grads exact)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("long_context on the GRPO prefix path — bitwise vs full-store")
    print("=" * 64)
    main()
