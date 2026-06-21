"""Muon+AdamW hybrid on the GRPO path (ResidentPrefixMoEModel — prefix-shared GRPO training step).

The prefix model inherits the hybrid wiring from ResidentMoEModel; this confirms the GRPO-specific
overrides (prefix _build_layer, the prefix MoE layer's own step) thread Muon correctly and that:
  (1) the prefix forward stays bitwise-DETERMINISTIC under muon (ratio=1 — the framework's core thesis,
      rollout==training; the optimizer must never perturb the forward);
  (2) GRPO POLICY IMPROVES under muon (rewarded completion's logprob rises over the others through
      fwd → grpo_loss_backward → Muon experts/proj + AdamW router/embed).
adamw is run alongside as the regression baseline.

Run:  python tests/training/test_muon_prefix.py
"""
import sys, os, time, copy
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


def run(cfg, w, prompt, comps, adv_g, Sp, Sc, G, optimizer, steps=25):
    pre = ResidentPrefixMoEModel(cfg, copy.deepcopy(w), Sp, Sc, G, optimizer=optimizer)
    # determinism: prefix forward twice → bitwise (ratio=1, optimizer-independent)
    h1 = pre.forward_prefix(prompt, comps, si); sync(); h1 = h1.copy()
    h2 = pre.forward_prefix(prompt, comps, si); sync()
    det = float(np.abs(h1.astype(np.float64) - h2.astype(np.float64)).max())

    ce, lp = pre.grpo_loss_backward(h2, comps, adv_g, si)
    gap0 = float(lp[0] - lp[1:].mean())
    t0 = time.perf_counter()
    for _ in range(steps):
        pre.step(si, lr=2e-3)
        h = pre.forward_prefix(prompt, comps, si)
        ce, lp = pre.grpo_loss_backward(h, comps, adv_g, si)
    sync(); dt = (time.perf_counter() - t0) / steps
    gap1 = float(lp[0] - lp[1:].mean())
    return dict(det=det, gap0=gap0, gap1=gap1, dt=dt)


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
    print("-" * 70)

    res = {}
    for opt in ("adamw", "muon"):
        res[opt] = run(cfg, w, prompt, comps, adv_g, Sp, Sc, G, opt)
        x = res[opt]
        print(f"  {opt:5s}: fwd det Δ={x['det']:.0e} ({'bitwise' if x['det']==0 else 'NONDET'})  "
              f"GRPO lp gap {x['gap0']:.1f}→{x['gap1']:.1f}  {x['dt']*1e3:.0f} ms/step")
    print("-" * 70)
    det = res["adamw"]["det"] == 0 and res["muon"]["det"] == 0
    impr = {o: res[o]["gap1"] > res[o]["gap0"] + 50.0 for o in res}
    print(f"  policy improvement: adamw={'OK' if impr['adamw'] else 'FAIL'}  muon={'OK' if impr['muon'] else 'FAIL'}")
    ok = det and impr["adamw"] and impr["muon"]
    print("=" * 70)
    print(f"  {'PASS — Muon hybrid threads the GRPO path: ratio=1 + policy improves' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Muon+AdamW hybrid — ResidentPrefixMoEModel (GRPO prefix-shared step)")
    print("=" * 70)
    main()
