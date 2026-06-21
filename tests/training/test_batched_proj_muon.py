"""BatchedProjMuon equivalence + convergence (task #24). Batching the per-layer proj Newton-Schulz
across same-shape weights (industrial Keller/Kimi pattern) must give the SAME update as the per-weight
NS (same math, just batched over the expert/weight grid) — within bf16 kernel tolerance — and SFT must
still converge. Compares optimizer="muon" with batch_proj=True (batched square proj) vs False (per-weight).
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
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
SQ = ["k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]   # the 1024² batched-square proj names


def run(cfg, w, ids, labels, STEPS, batch_proj):
    m = ResidentMoEModel(cfg, copy.deepcopy(w), 2, ids.shape[1], device_route=True,
                         optimizer="muon", batch_proj=batch_proj)
    ces = []
    for _ in range(STEPS):
        h = m.forward(ids, si)
        ces.append(m.loss_backward(h, labels, si))
        m.step(si, lr=2e-3); sync()
    # read the square proj weights after training (bf16 → f32)
    wts = {}
    for j in range(cfg.n_layers):
        for n in SQ:
            if n in m.layers[j].w:
                wts[f"{n}{j}"] = b2f(m.layers[j].w[n].to_numpy())
    return ces, wts


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)
    B, S = 2, 128
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, B, S)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, (B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (B * S,)).astype(np.int64)

    ces_p, wts_p = run(cfg, w, ids, labels, 6, batch_proj=False)   # per-weight reference
    ces_b, wts_b = run(cfg, w, ids, labels, 6, batch_proj=True)    # batched

    print("-" * 64)
    worst_c, worst_r = 1.0, 0.0; ncmp = 0
    for k in wts_p:
        a, b = wts_b[k].ravel(), wts_p[k].ravel()
        if np.linalg.norm(b) < 1e-6:                 # skip MoE layers' DUMMY gate/up/down (all zeros)
            continue
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        r = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
        worst_c, worst_r = min(worst_c, c), max(worst_r, r); ncmp += 1
    # batched NS == per-weight NS within bf16 (same per-weight math, batched over the grid)
    eq = worst_c > 0.9999 and worst_r < 0.01
    conv = ces_b[-1] < 0.5 * ces_b[0]
    print(f"  weights batched-vs-perweight ({ncmp} real proj): worst cos {worst_c:.5f}  rel {worst_r:.2%}  {'OK' if eq else 'FAIL'}")
    print(f"  CE: per-weight {ces_p[0]:.3f}→{ces_p[-1]:.3f}   batched {ces_b[0]:.3f}→{ces_b[-1]:.3f}  "
          f"{'OK (converges)' if conv else 'FAIL'}")
    print("=" * 64)
    print(f"  {'PASS — batched proj Muon == per-weight (within bf16) + converges' if (eq and conv) else 'FAIL'}")
    return eq and conv


if __name__ == "__main__":
    print("BatchedProjMuon — batched square proj NS vs per-weight")
    print("=" * 64)
    main()
