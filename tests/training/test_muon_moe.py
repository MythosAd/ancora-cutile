"""Muon+AdamW hybrid on the MoE family (ResidentMoEModel — the 仿造MAI dense/MoE + local/global model).

Extends the dense-model hybrid to where the MoE optimizer memory actually lives: the 3D EXPERT weights
(Wg/Wu/Wd, E square 1024×1024 experts/MoE-layer) → BATCHED resident Muon (one NS chain over all E
experts, Kimi/Moonshot recipe). 2D attention proj → resident Muon; tied embed/LM-head + 1D gains +
MoE router → AdamW. Muon keeps one momentum buffer vs AdamW's m+v → drops the experts' v (the FFN's
optimizer bulk).

Validates: (1) forward bitwise-DETERMINISTIC under muon (ratio=1 untouched); (2) SFT CE collapses
under muon (adamw alongside as the regression baseline); (3) optimizer-state VRAM measurably lower;
and reports the step-time cost of the expert NS (honest — Muon-on-experts pays the GEMM ceiling).

Run:  python tests/training/test_muon_moe.py [n_layers] [n_steps]
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
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
def memfree(): sync(); return cudart.cudaMemGetInfo()[1]


def run(cfg, w, ids, labels, ce0_ref, STEPS, optimizer):
    f0 = memfree()                                            # before construction
    model = ResidentMoEModel(cfg, copy.deepcopy(w), cfg._B, cfg._S, optimizer=optimizer)

    h1 = model.forward(ids, si); sync(); h1 = h1.copy()       # determinism BEFORE any weight update
    h2 = model.forward(ids, si); sync()
    det = np.abs(h1.astype(np.float64) - h2.astype(np.float64)).max()

    ce0 = None; cemin = 1e9; dts = []; used = None
    for it in range(STEPS):
        t = time.perf_counter()
        h = model.forward(ids, si)
        ce = model.loss_backward(h, labels, si)
        model.step(si, lr=2e-3)                               # step 0 LAZILY allocs the optimizer state
        sync()
        dts.append(time.perf_counter() - t)
        if it == 0: ce0 = ce; used = (f0 - memfree()) / 1e9   # measure AFTER opt state exists (model+opt)
        cemin = min(cemin, ce)
    return dict(used=used, det=det, ce0=ce0, cemin=cemin, dt=np.mean(dts[1:]), model=model)


def main(NL=4, STEPS=100):
    cfg = MoEConfig(vocab=2048, n_layers=NL, period=2)         # interleave (L,D)(G,M)…
    B, S = 2, 128; cfg._B, cfg._S = B, S; M = B * S
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    print("schedule (is_global, ffn_dense):", host.sched)
    n_moe = sum(1 for s in host.sched if not s[1])
    w = from_host(host, B, S)

    # analytic v-buffer saving: experts dominate. Per MoE layer 3 tensors × E·H·Ie params; Muon drops v.
    E, H, Ie = cfg.n_experts, cfg.hidden, cfg.expert_inter
    exp_v = n_moe * 3 * E * H * Ie * 4 / 1e9
    print(f"  {n_moe} MoE layers × 3 × E·H·Ie = expert v-buffer {exp_v:.2f} GB dropped (+ proj v); "
          f"shared NS scratch ≈ {(5*E*H*Ie*2 + 1)/1e9:.2f} GB expert + ~0.02 GB proj")
    print("-" * 70)

    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, (B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (M,)).astype(np.int64)

    r = {}
    for opt in ("adamw", "muon"):
        r[opt] = run(cfg, w, ids, labels, None, STEPS, opt)
        x = r[opt]
        print(f"  {opt:5s}: VRAM {x['used']:.2f} GB  det={'bitwise' if x['det']==0 else 'NONDET'}  "
              f"CE {x['ce0']:.3f}→min {x['cemin']:.3f} ({x['cemin']/x['ce0']*100:.0f}%)  {x['dt']*1e3:.0f} ms/step")
    print("-" * 70)

    drop = r["adamw"]["used"] - r["muon"]["used"]
    coll = {o: r[o]["cemin"] < 0.5 * r[o]["ce0"] for o in r}
    det = r["adamw"]["det"] == 0 and r["muon"]["det"] == 0
    print(f"  measured VRAM drop muon vs adamw: {drop:.2f} GB  (analytic expert v-buffer {exp_v:.2f} GB)")
    print(f"  CE collapse: adamw={'OK' if coll['adamw'] else 'FAIL'}  muon={'OK' if coll['muon'] else 'FAIL'}")
    print(f"  step cost of expert NS: {(r['muon']['dt']-r['adamw']['dt'])*1e3:.0f} ms/step "
          f"({r['adamw']['dt']*1e3:.0f}→{r['muon']['dt']*1e3:.0f})")
    ok = coll["adamw"] and coll["muon"] and det and drop > 0
    print("=" * 70)
    print(f"  {'PASS — Muon on MoE experts trains + drops optimizer VRAM' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print("Muon+AdamW hybrid — ResidentMoEModel (MoE experts → batched Muon)")
    print("=" * 70)
    main(NL, STEPS)
