"""Muon+AdamW hybrid on the device-resident ResidentModel (real Qwen3-0.6B).

The hybrid: 2D PROJ matrices (q/k/v/o/gate/up/down) → resident Muon (momentum-only state, ONE shared
NS scratch); the tied embed/LM-head + 1D RMSNorm gains → AdamW (Muon there hurts — [[mfu-strategy]]).
Muon keeps ONE momentum buffer vs AdamW's m+v → it DROPS the per-PROJ `v` (4 B/param ≈ 1.7 GB on
Qwen3-0.6B), lowering the optimizer-state VRAM floor (the long_context lever).

Validates: (1) forward stays bitwise-DETERMINISTIC under the muon build (optimizer must not perturb
the ratio=1 forward); (2) SFT overfit CE COLLAPSES under muon (the hybrid actually optimizes), with
adamw run alongside as the regression baseline; (3) the optimizer-state VRAM is measurably LOWER
under muon, matching the analytic v-buffer saving.

Run:  python tests/training/test_muon_hybrid.py [n_layers] [n_steps]
"""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.load_qwen3 import load_qwen3

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def memfree():
    cudart.cudaStreamSynchronize(si); gc.collect()
    return cudart.cudaMemGetInfo()[1]          # (err, free, total) → free bytes


def run(weights, cfg, V, NL, STEPS, optimizer):
    B, S = 1, 128
    rng = np.random.default_rng(1)
    ids = rng.integers(0, V, (B, S)).astype(np.int64)
    labels = rng.integers(0, V, B * S).astype(np.int64)

    f0 = memfree()
    rm = ResidentModel(cfg, weights, B, S, V, optimizer=optimizer)
    used = (f0 - memfree()) / 1e9              # GB the whole model+optimizer footprint took

    # determinism (forward twice → bitwise): the optimizer state must not touch the forward
    d1 = rm.forward(ids, si).copy(); d2 = rm.forward(ids, si).copy()
    det = np.array_equal(d1, d2)

    ces, dt = [], []
    for step in range(STEPS):
        t = time.time()
        h = rm.forward(ids, si)
        ce = rm.loss_backward(h, labels, si)
        rm.step(si, lr=2e-3)                    # AdamW lr=2e-3 (embed/norm); PROJ Muon uses muon_lr=0.02
        ces.append(ce); dt.append(time.time() - t)
    rm.free(); gc.collect()
    return dict(used=used, det=det, ces=ces, dt=np.mean(dt))


def main(NL=6, STEPS=10):
    cfg = Qwen3Config(); V = 151936
    print(f"loading REAL Qwen3-0.6B ({NL} of 28 layers)...")
    t0 = time.time(); weights = load_qwen3(n_layers=NL); print(f"  loaded in {time.time()-t0:.0f}s")

    # analytic v-buffer saving: AdamW keeps m+v per PROJ param; Muon keeps buf only → drops v (4 B/param)
    per_layer = sum(int(np.prod(weights["layers"][0][n].shape)) for n in PROJ)
    save_nl = per_layer * NL * 4 / 1e9
    save_28 = per_layer * 28 * 4 / 1e9
    print(f"  PROJ params/layer = {per_layer/1e6:.1f}M → v-buffer drop {save_nl:.2f} GB @ {NL}L, "
          f"{save_28:.2f} GB @ 28L")
    print("-" * 70)

    r = {}
    for opt in ("adamw", "muon"):
        r[opt] = run(weights, cfg, V, NL, STEPS, opt)
        c = r[opt]["ces"]
        print(f"  {opt:5s}: VRAM {r[opt]['used']:.2f} GB  det={'OK' if r[opt]['det'] else 'FAIL'}  "
              f"CE {c[0]:.2f}→{c[-1]:.2f} (best {min(c):.2f})  {r[opt]['dt']:.2f}s/step")
    print("-" * 70)

    drop = r["adamw"]["used"] - r["muon"]["used"]
    coll = {o: min(r[o]["ces"]) < r[o]["ces"][0] - 3.0 for o in r}          # clear multi-nat collapse
    det = r["adamw"]["det"] and r["muon"]["det"]
    mem_ok = drop > 0.5 * save_nl                                           # most of the v-buffer recovered
    print(f"  measured VRAM drop muon vs adamw: {drop:.2f} GB  (analytic v-buffer {save_nl:.2f} GB @ {NL}L)")
    print(f"  CE collapse: adamw={'OK' if coll['adamw'] else 'FAIL'}  muon={'OK' if coll['muon'] else 'FAIL'}")
    ok = coll["adamw"] and coll["muon"] and det and mem_ok
    print("=" * 70)
    print(f"  {'PASS — Muon hybrid trains + drops optimizer VRAM' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print("Muon+AdamW hybrid — ResidentModel real Qwen3-0.6B")
    print("=" * 70)
    main(NL, STEPS)
