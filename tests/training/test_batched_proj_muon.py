"""BatchedProjMuon equivalence + convergence (tasks #24/#30). Batching the per-layer proj Newton-
Schulz across same-NS-shape weights (industrial Keller/Kimi pattern) must give the SAME update as the
per-weight NS (same math, just batched over the expert/weight grid) — within bf16 kernel tolerance —
and SFT must still converge. Since #30 ALL 2D proj weights batch: square k/v (+dense gate/up/down,
1024²) AND the rectangular q/o ((1024,2048) NS shape; o transposed in/out by the fused _muon_mom_t/
_muon_update_cast_t). Also gates the pipeline-parallel granularity: muon_scope=1 (per-layer batches)
must be BITWISE == global batches — the per-matrix NS math is independent of the batch grid E.
Compares optimizer="muon" with batch_proj=True (batched) vs False (per-weight reference)."""
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
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # all 2D batched now


def run(cfg, w, ids, labels, STEPS, batch_proj, scope=None):
    m = ResidentMoEModel(cfg, copy.deepcopy(w), 2, ids.shape[1], device_route=True,
                         optimizer="muon", batch_proj=batch_proj, muon_scope=scope)
    ces = []
    for _ in range(STEPS):
        h = m.forward(ids, si)
        ces.append(m.loss_backward(h, labels, si))
        m.step(si, lr=2e-3); sync()
    # read the proj weights after training (bf16 bits — keep bits for the bitwise scope gate)
    wts = {}
    for j in range(cfg.n_layers):
        for n in PROJ:
            if n in m.layers[j].w:
                wts[f"{n}{j}"] = m.layers[j].w[n].to_numpy().copy()
    return ces, wts


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)
    B, S = 2, 128
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, B, S)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, (B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (B * S,)).astype(np.int64)

    ces_p, wts_p = run(cfg, w, ids, labels, 6, batch_proj=False)          # per-weight reference
    ces_b, wts_b = run(cfg, w, ids, labels, 6, batch_proj=True)           # batched, global scope
    ces_1, wts_1 = run(cfg, w, ids, labels, 6, batch_proj=True, scope=1)  # batched, per-layer scope

    print("-" * 64)
    worst_c, worst_r, worst = 1.0, 0.0, ""; ncmp = 0
    for k in wts_p:
        a, b = b2f(wts_b[k]).ravel(), b2f(wts_p[k]).ravel()
        if np.linalg.norm(b) < 1e-6:                 # skip MoE layers' DUMMY gate/up/down (all zeros)
            continue
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        r = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
        if c < worst_c: worst = k
        worst_c, worst_r = min(worst_c, c), max(worst_r, r); ncmp += 1
    # batched NS == per-weight NS within bf16 (same per-weight math, batched over the grid)
    eq = worst_c > 0.9999 and worst_r < 0.01
    conv = ces_b[-1] < 0.5 * ces_b[0]
    # per-layer scope must be BITWISE == global scope (grouping only changes the launch grid)
    bit = all((wts_1[k] == wts_b[k]).all() for k in wts_b)
    print(f"  weights batched-vs-perweight ({ncmp} proj incl q/o): worst cos {worst_c:.5f} ({worst})  "
          f"rel {worst_r:.2%}  {'OK' if eq else 'FAIL'}")
    print(f"  per-layer scope (muon_scope=1) vs global: {'BITWISE ==' if bit else 'DIFFERS'}  "
          f"{'OK' if bit else 'FAIL'}")
    print(f"  CE: per-weight {ces_p[0]:.3f}→{ces_p[-1]:.3f}   batched {ces_b[0]:.3f}→{ces_b[-1]:.3f}  "
          f"{'OK (converges)' if conv else 'FAIL'}")
    ok = eq and conv and bit
    print("=" * 64)
    print(f"  {'PASS — batched Muon (ALL 2D proj) == per-weight + scope-invariant + converges' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("BatchedProjMuon — batched NS over ALL 2D proj (square + rect q/o) vs per-weight")
    print("=" * 64)
    main()
