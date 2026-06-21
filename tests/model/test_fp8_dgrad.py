"""FP8 E4M3 data-gradient (dgrad) on the MoE family — the MAI/DeepSeek backward precision.
  (A) fp8_bwd is FORWARD-TRANSPARENT: a fp8_bwd model's forward is BITWISE-identical to the BF16
      model's (dgrad is backward-only → not in the rollout forward → ratio=1 UNTOUCHED). THE gate.
  (B) SFT overfit still collapses with the FP8 dgrad (the QAT-style straight-through trains).
  (C) grads differ from BF16 by a few % that COMPOUNDS down the layers (the dgrad quant signature,
      not a bug): top-layer dW ≈ BF16 (its dy is the identical boundary), bottom layers drift more.
  (D) forward determinism bitwise."""
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


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)
    G, S = 4, 128
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, size=(G, S)).astype(np.int64)
    labels = np.concatenate([ids[:, 1:], np.zeros((G, 1), np.int64)], 1)

    bf = ResidentMoEModel(cfg, from_host(host, G, S), G, S, device_route=True, fp8_bwd=False)
    fp = ResidentMoEModel(cfg, from_host(host, G, S), G, S, device_route=True, fp8_bwd=True)

    # ── (A) forward TRANSPARENT: fp8_bwd forward == bf16 forward bitwise (ratio=1 untouched) ──
    hb = bf.forward(ids, si).copy()
    hf = fp.forward(ids, si).copy()
    eA = int(np.abs(hb.astype(np.int32) - hf.astype(np.int32)).max())
    print(f"  (A) forward fp8_bwd vs BF16: Δ={eA}  "
          f"{'OK (forward-transparent → ratio=1 untouched)' if eA == 0 else 'FAIL'}")

    # ── (D) forward determinism ──
    hf2 = fp.forward(ids, si).copy()
    eD = int(np.abs(hf2.astype(np.int32) - hf.astype(np.int32)).max())
    print(f"  (D) fp8_bwd forward determinism: Δ={eD}  {'OK (bitwise)' if eD == 0 else 'FAIL'}")

    # ── (C) one backward each → the FP8-dgrad grad must point the SAME WAY as BF16 (cosine≈1):
    #        a few-% quant noise preserves DIRECTION; a wiring bug (wrong scale/index) would not. ──
    bf.forward(ids, si); bf.loss_backward(None, labels.reshape(-1), si)
    fp.forward(ids, si); fp.loss_backward(None, labels.reshape(-1), si)
    def cos(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    mrel = lambda a, b: float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-30))
    A, Bp = fp.layers[-1].G["q_proj"].to_numpy(), bf.layers[-1].G["q_proj"].to_numpy()
    C, D = fp.layers[0].G["q_proj"].to_numpy(), bf.layers[0].G["q_proj"].to_numpy()
    fin = np.all(np.isfinite(A)) and np.all(np.isfinite(C))
    ct_, cb_ = cos(A, Bp), cos(C, D)
    okC = fin and ct_ > 0.9 and cb_ > 0.9           # well-aligned with BF16 ⇒ quant noise, not a bug
    print(f"  (C) grad vs BF16 (q_proj): cosine top {ct_:.4f} / bottom {cb_:.4f}  "
          f"mean-rel {mrel(A,Bp):.1%}/{mrel(C,D):.1%}  "
          f"{'OK (same direction → quant noise, not a bug)' if okC else 'FAIL'}")

    # ── (B) SFT overfit collapses under the FP8 dgrad ──
    def overfit(m, n=120):
        c0 = c = None
        for t in range(n):
            m.forward(ids, si); c = m.loss_backward(None, labels.reshape(-1), si)
            if t == 0: c0 = c
            m.step(si, lr=2e-3); cudart.cudaStreamSynchronize(si)
        return c0, c
    cb0, cb = overfit(bf); cf0, cf = overfit(fp)
    okB = cf < 0.1 * cf0
    print(f"  (B) SFT overfit: BF16 {cb0:.2f}→{cb:.4f} | FP8-dgrad {cf0:.2f}→{cf:.4f}  "
          f"{'OK (FP8 dgrad trains)' if okB else 'FAIL'}")
    return eA == 0 and eD == 0 and okC and okB


if __name__ == "__main__":
    ok = main()
    print("  PASS" if ok else "  FAIL")
