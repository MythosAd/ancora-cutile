"""GRADIENT ACCUMULATION equivalence: a B=2 single step vs B=1 × 2 accumulated micro-batches
(same two sequences, same norm) must produce
  (A) per-sequence logprobs BITWISE identical (batch invariance — rows are independent),
  (B) every weight grad equal up to reduction-regrouping ulp (Σ over 512 rows vs 256+256),
  (C) determinism: the accumulated path twice → bitwise-identical grads.
Then one AdamW step on both and the updated weights compared."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

REL = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def grads(m):
    g = {"embed": m.gegrad.to_numpy(), "fng": m.gfng.to_numpy()}
    for i, l in enumerate(m.layers):
        g[f"L{i}.q_proj"] = l.G["q_proj"].to_numpy()
        g[f"L{i}.input_ln"] = l.G["input_ln"].to_numpy()
        if hasattr(l, "moe"):
            g[f"L{i}.dWd"] = l.moe.dWd.to_numpy()
            g[f"L{i}.router"] = l.moe.Gr_acc.to_numpy()
        else:
            g[f"L{i}.down_proj"] = l.G["down_proj"].to_numpy()
    return g


def run(S):
    """B=2 single step vs B=1×2 accumulated, at sequence length S. At S=768 the B=2 model is
    M=1536 > MC=1024 → the boundary runs as 2 CHUNKS (a sequence straddles the chunk seam),
    while each B=1 micro-batch is M=768 (1 chunk). So this also validates the chunked-boundary
    cross-chunk dW accumulation (the M>1024 path the small tests never reach)."""
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)
    Mtot = 2 * S
    nchunk = (2 * S + 1023) // 1024
    print(f"  S={S} (B=2 model M={2*S} → {nchunk} boundary chunk(s); B=1 micro-batch M={S} → 1 chunk):")
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, size=(2, S)).astype(np.int64)
    labels = np.concatenate([ids[:, 1:], np.zeros((2, 1), np.int64)], 1)

    # ── reference: B=2 single step ──
    A = ResidentMoEModel(cfg, from_host(host, 2, S), 2, S, device_route=True)
    A.forward(ids, si)
    ceA = A.loss_backward(None, labels.reshape(-1), si, norm=Mtot)
    lpA = A.glp.to_numpy().reshape(2, S)
    gA = grads(A)

    # ── accumulated: B=1, two micro-batches ──
    B = ResidentMoEModel(cfg, from_host(host, 1, S), 1, S, device_route=True)
    def acc_pass():
        B.forward(ids[:1], si)
        ce0 = B.loss_backward(None, labels[0], si, norm=Mtot, accumulate=False)
        lp0 = B.glp.to_numpy().reshape(S).copy()
        B.forward(ids[1:], si)
        ce1 = B.loss_backward(None, labels[1], si, norm=Mtot, accumulate=True)
        lp1 = B.glp.to_numpy().reshape(S).copy()
        return ce0 + ce1, np.stack([lp0, lp1]), grads(B)
    ceB, lpB, gB = acc_pass()

    eLP = float(np.abs(lpA - lpB).max())
    print(f"  (A) per-sequence lp: B=2 vs accumulated  Δ={eLP:.0e}  {'OK (bitwise)' if eLP == 0 else 'FAIL'}")
    print(f"      ce: {ceA:.6f} vs {ceB:.6f}  Δ={abs(ceA-ceB):.1e}")
    worst = ("", 0.0)
    for k in gA:
        r = REL(gB[k], gA[k])
        if r > worst[1]: worst = (k, r)
    okB = worst[1] < 2e-3
    print(f"  (B) grads vs B=2: worst {worst[0]} rel={worst[1]:.2e}  "
          f"{'OK (reduction-regroup ulp)' if okB else 'FAIL'}")

    ceB2, lpB2, gB2 = acc_pass()
    eD = max(float(np.abs(gB2[k] - gB[k]).max()) for k in gB)
    print(f"  (C) accumulated path repeat-determinism: grad Δ={eD:.0e}  {'OK (bitwise)' if eD == 0 else 'FAIL'}")

    # ── one AdamW step on both → compare an updated weight (bf16 bits via fresh forward) ──
    A.step(si, lr=1e-3); B.step(si, lr=1e-3)
    cudart.cudaStreamSynchronize(si)
    wA = A.layers[0].w["q_proj"].to_numpy(); wB = B.layers[0].w["q_proj"].to_numpy()
    nW = float((wA != wB).mean())
    print(f"  (D) post-AdamW q_proj weights: {100*(1-nW):.2f}% bits identical "
          f"(rest = the grad-ulp through AdamW)  {'OK' if nW < 0.02 else 'FAIL'}")
    A.free() if hasattr(A, "free") else None
    return eLP == 0 and okB and eD == 0 and nW < 0.02


def main():
    ok = True
    for S in (256, 768):                  # 256 = single chunk; 768 = M=1536 → 2 chunks (multi-chunk path)
        ok = run(S) and ok
    return ok


if __name__ == "__main__":
    ok = main()
    print("  PASS" if ok else "  FAIL")
