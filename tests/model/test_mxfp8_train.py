"""MXFP8 training closed loop — the forward GEMMs run in MXFP8, the backward in BF16, and the AdamW
-updated weights are RE-QUANTIZED to fp8 each step so the forward tracks the updates. Validates:

  (1) MXFP8 training overfit (fwd MXFP8 → bwd BF16 → AdamW → device weight re-quant) COLLAPSES the loss.
  (2) CONTROL: freeze the fp8 forward weights (skip re-quant) → loss does NOT collapse — i.e. the
      device weight re-quant is what closes the loop (without it the forward never sees the updates).
  (3) BF16 training still collapses (regression: mxfp8=False unchanged).

Run:  python tests/model/test_mxfp8_train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_train import ResidentLayerTrain

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden; B, S = 1, 128; M = B * S
bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def run(mxfp8, requant=True, steps=30, lr=2e-3):
    """fwd→bwd→AdamW on a fixed MSE target; returns the loss trace."""
    layer = TransformerLayer(cfg, seed=2); rng = np.random.default_rng(3)
    x = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    y = _bf((rng.standard_normal((B, S, H)) * 0.5).astype(np.float32))
    tl = ResidentLayerTrain(cfg, layer.w, B, S, mxfp8=mxfp8)
    if mxfp8 and not requant:                       # CONTROL: quant once, then freeze (re-quant = no-op)
        tl._requant_w(si)
        tl._requant_w = lambda s: setattr(tl, "_wq_dirty", False)
    gx = _DBuf(np.ascontiguousarray(x.reshape(M, H), np.float32)); gd = _DBuf.zeros((M, H), np.uint16)
    yt = bf32(_f32bf(y.reshape(M, H)))
    losses = []
    for _ in range(steps):
        out = tl.forward(gx, si); cudart.cudaStreamSynchronize(si)
        o = out.to_numpy()
        losses.append(float(np.mean((o - yt) ** 2)))
        cdrv.cuMemcpyHtoD(gd.ptr, np.ascontiguousarray(_f32bf((2.0 / M) * (o - yt))), gd.nbytes)
        tl.backward(gd, si); tl.step(si, lr=lr)
    return losses


def main():
    print("MXFP8 training closed loop — fwd MXFP8 / bwd BF16 / device weight re-quant"); print("=" * 74)
    mx   = run(mxfp8=True,  requant=True)
    frz  = run(mxfp8=True,  requant=False)
    bf16 = run(mxfp8=False)

    def report(tag, L):
        ok = L[-1] < 0.5 * L[0]
        print(f"  {tag:24s} loss {L[0]:.4f} → {min(L):.4f}  ({min(L)/L[0]*100:5.1f}% of init)  "
              f"{'OK — collapses' if ok else 'FLAT — no learning'}")
        return ok, min(L)

    c1, mx_best   = report("(1) MXFP8 + re-quant", mx)
    c2, frz_best  = report("(2) MXFP8 frozen fp8", frz)
    c3, _         = report("(3) BF16 (regression)", bf16)
    # re-quant must collapse; frozen must clearly NOT collapse as much (re-quant is load-bearing)
    requant_matters = mx_best < 0.3 * frz_best
    print(f"  re-quant load-bearing: MXFP8+re-quant best {mx_best:.4f}  <<  frozen best {frz_best:.4f}  "
          f"{'OK' if requant_matters else 'FAIL'}")

    ok = c1 and (not c2) and c3 and requant_matters
    print("=" * 74)
    print(f"  {'PASS — MXFP8 training closed loop works (re-quant tracks weight updates)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
