"""MXFP8 forward — rollout==training BITWISE, and accuracy vs BF16.

The precision recipe wants the forward GEMMs in MXFP8 (not BF16). For on-policy RL that's only safe if
moving to MXFP8 KEEPS the rollout↔training bitwise match (ratio=1). MXFP8 quant is per-row-per-32-block
and mma_scaled is fixed-K/no-split-K → both per-row → the same MGEMM=128 full-tile trick that made the
BF16 decode bitwise-equal to prefill carries over. This test proves it:

  (A) MXFP8 decode (rollout) hidden[t]  ==  MXFP8 prefill (training) hidden[t]   BITWISE, every position
      ⇒ ratio π_train/π_infer = 1 EXACTLY under MXFP8 (the goal: unify rollout & training forward precision).
  (B) MXFP8 forward vs BF16 forward accuracy drift (sanity — MXFP8 is lower precision, expect a few %).

Run:  python tests/model/test_mxfp8_unified.py [n_layers]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_train import ResidentLayerTrain
from ancora.model.resident_decode import ResidentDecodeLayer, MGEMM

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden
ub = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def _free_obj(objs):
    seen = set()
    def visit(o):
        if isinstance(o, _DBuf):
            if int(o.ptr) not in seen:
                seen.add(int(o.ptr)); o.free()
        elif isinstance(o, dict):
            for v in o.values(): visit(v)
        elif isinstance(o, (list, tuple)):
            for v in o: visit(v)
    for o in objs:
        visit(o.__dict__ if hasattr(o, "__dict__") else o)


def prefill(layers_w, x0_all, S, mxfp8):
    """Training prefill (whole sequence at once) → hidden (S,H) f32."""
    layers = [ResidentLayerTrain(cfg, w, 1, S, sr_grad=False, mxfp8=mxfp8) for w in layers_w]
    gin0 = _DBuf(np.ascontiguousarray(x0_all, np.float32)); gx = gin0
    for l in layers: gx = l.forward(gx, si)
    cudart.cudaStreamSynchronize(si)
    out = gx.to_numpy().copy()
    gin0.free(); _free_obj(layers)
    return out


def decode_rollout(layers_w, x0_all, S, maxS, mxfp8):
    """Rollout decode (one token at a time, per-layer KV-cache) → assembled hidden (S,H) f32."""
    Md = MGEMM
    layers = [ResidentDecodeLayer(cfg, w, 1, maxS, mxfp8=mxfp8) for w in layers_w]
    gin = _DBuf.zeros((Md, H), np.float32); dhid = _DBuf.zeros((S, H), np.float32)
    x0_rows = [np.ascontiguousarray(x0_all[t], np.float32) for t in range(S)]
    for t in range(S):
        cdrv.cuMemcpyHtoDAsync(gin.ptr, x0_rows[t], H * 4, si)
        x = gin
        for l in layers: x = l.forward(x, t, si)
        cdrv.cuMemcpyDtoDAsync(int(dhid.ptr) + t * H * 4, int(x.ptr), H * 4, si)
    cudart.cudaStreamSynchronize(si)
    out = dhid.to_numpy().copy()
    gin.free(); dhid.free(); _free_obj(layers)
    return out


def main(NL=2, real=False):
    S = 128
    rng = np.random.default_rng(0)
    if real:   # REAL Qwen3-0.6B weights → credible MXFP8-vs-BF16 accuracy (synthetic random is a hard case)
        from ancora.model.load_qwen3 import load_qwen3
        w = load_qwen3(n_layers=NL); layers_w = w["layers"]; embed = w["embed"].astype(np.float32); V = embed.shape[0]
    else:
        V = 2048
        layers_w = [TransformerLayer(cfg, seed=i).w for i in range(NL)]
        embed = (rng.standard_normal((V, H)) * 0.02).astype(np.float32)
    print(f"MXFP8 forward — rollout==training bitwise + accuracy vs BF16   NL={NL} S={S} V={V} "
          f"{'REAL Qwen3-0.6B' if real else 'synthetic'}")
    print("=" * 80)
    ids = rng.integers(0, V, S).astype(np.int64)
    eb = _f32bf(embed)
    x0_all = (eb[ids].astype(np.uint32) << 16).view(np.float32).astype(np.float32)   # (S,H) embed gather

    t0 = time.time(); pre_mx = prefill(layers_w, x0_all, S, mxfp8=True)
    dec_mx = decode_rollout(layers_w, x0_all, S, S, mxfp8=True)
    pre_bf = prefill(layers_w, x0_all, S, mxfp8=False)
    print(f"  ran prefill(MXFP8) + decode(MXFP8) + prefill(BF16)  ({time.time()-t0:.1f}s)")
    print("-" * 80)

    # ── (A) MXFP8 rollout == MXFP8 training, bitwise ──
    same = (ub(pre_mx) == ub(dec_mx)).all(axis=1)
    nbad = int((~same).sum())
    print(f"  (A) MXFP8 decode == MXFP8 prefill bitwise: {S-nbad}/{S} positions, max|Δ|={np.abs(pre_mx-dec_mx).max():.3g}  "
          f"{'OK' if nbad == 0 else 'FAIL'}")
    if nbad:
        for t in np.where(~same)[0][:6]:
            print(f"        pos {t:3d}: max|Δ|={np.abs(pre_mx[t]-dec_mx[t]).max():.3g}")

    # ── (B) MXFP8 forward accuracy vs BF16 forward (INFORMATIONAL — synthetic random weights are a
    #    hard case for block-scaling; real Qwen3 activations are far kinder, cf. ResidentLayer ~4.6%/layer).
    #    A catastrophe guard (broken kernel) is the only hard fail here; PASS rests on (A) bitwise. ──
    drift = rel(pre_mx, pre_bf)
    print(f"  (B) MXFP8 forward vs BF16 forward [{'REAL' if real else 'synthetic'}, informational]: "
          f"rel max|Δ|={drift*100:.2f}%  per-elem mean {np.abs(pre_mx-pre_bf).mean()/(np.abs(pre_bf).mean()+1e-9)*100:.2f}%  "
          f"({'sane' if drift < 0.5 else 'BROKEN?'})")

    ok = (nbad == 0) and (drift < 0.5)
    print("=" * 80)
    print(f"  {'PASS — MXFP8 forward unifies rollout & training (ratio=1 under MXFP8)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    real = "real" in sys.argv[2:]
    sys.exit(0 if main(NL, real) else 1)
