"""ResidentDecodeModel — full device-resident ROLLOUT engine, logprobs BITWISE vs ResidentModel (training).

The on-policy RL invariant: the logprob the rollout engine reports for a token must equal the logprob
the TRAINING forward assigns to that same token (ratio π_train/π_infer = 1). This test proves it at the
FULL-MODEL level (tied embed/LM-head boundary included), against the actual training engine ResidentModel:

  (1) TEACHER-FORCED:  dm.score(ids, labels) per-token logprob  ==  ResidentModel(ids).logprob   bitwise
  (2) GREEDY GENERATE: dm.generate(prompt, n_new) per-token rollout logprob  ==  ResidentModel(generated).logprob
                       bitwise — i.e. the logprob produced DURING generation equals what training assigns.

Run:  python tests/model/test_resident_decode_model.py [n_layers]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.resident_decode import ResidentDecodeModel

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden
bits = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32)


def train_logprob(weights, V, ids, labels):
    """ResidentModel (training) per-token logprob for `ids` scored against `labels`."""
    S = ids.size
    rm = ResidentModel(cfg, weights, 1, S, V)
    h = rm.forward(ids.reshape(1, S), si)
    rm.loss_backward(h, labels, si)                       # fills rm.glp (the per-token logprob)
    lp = rm.glp.to_numpy().reshape(S).copy()
    rm.free()
    return lp


def main(NL=2):
    P, n_new, V = 64, 64, 2048
    S = P + n_new
    print(f"ResidentDecodeModel — rollout logprob vs ResidentModel training   NL={NL} P={P} n_new={n_new} V={V}")
    print("=" * 84)
    rng = np.random.default_rng(0)
    weights = {"layers": [TransformerLayer(cfg, seed=i).w for i in range(NL)],
               "embed": (rng.standard_normal((V, H)) * 0.02).astype(np.float32),
               "final_norm": (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)}
    ids = rng.integers(0, V, S).astype(np.int64)
    labels = np.concatenate([ids[1:], [0]]).astype(np.int64)        # next-token; last is a dummy

    # ── (1) teacher-forced score vs training ──
    t0 = time.time(); train_lp = train_logprob(weights, V, ids, labels)
    dm = ResidentDecodeModel(cfg, weights, 1, S, V)
    score_lp = dm.score(ids, labels, si); dm.free()
    same1 = np.array_equal(bits(score_lp), bits(train_lp))
    print(f"  (1) teacher-forced: rollout score == training logprob   max|Δ|={np.abs(score_lp-train_lp).max():.3g}  "
          f"mean CE={float(-train_lp.mean()):.4f}  {'OK' if same1 else 'FAIL'}  ({time.time()-t0:.1f}s)")
    if not same1:
        bad = np.where(~(bits(score_lp) == bits(train_lp)))[0]
        for t in bad[:6]:
            print(f"        pos {t:3d}: rollout={score_lp[t]:.6f} train={train_lp[t]:.6f}")

    # ── (2) greedy generation: the logprob produced DURING rollout == training's logprob ──
    t0 = time.time()
    dm2 = ResidentDecodeModel(cfg, weights, 1, S, V)
    gen_ids, gen_lp = dm2.generate(ids[:P], n_new, si); dm2.free()
    full = np.concatenate([ids[:P], gen_ids]).astype(np.int64)
    full_labels = np.concatenate([full[1:], [0]]).astype(np.int64)
    train_lp2 = train_logprob(weights, V, full, full_labels)
    tgt = train_lp2[P - 1: P - 1 + n_new]                          # training logprob of each generated token
    same2 = np.array_equal(bits(gen_lp), bits(tgt))
    print(f"  (2) greedy generate: rollout logprob == training logprob   max|Δ|={np.abs(gen_lp-tgt).max():.3g}  "
          f"({n_new} tokens)  {'OK' if same2 else 'FAIL'}  ({time.time()-t0:.1f}s)")
    if not same2:
        bad = np.where(~(bits(gen_lp) == bits(tgt)))[0]
        for i in bad[:6]:
            print(f"        gen[{i}] tok={gen_ids[i]:5d}: rollout={gen_lp[i]:.6f} train={tgt[i]:.6f}")

    ok = same1 and same2
    print("=" * 84)
    print(f"  {'PASS — rollout engine logprobs are BITWISE-equal to training (ratio=1 exactly)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sys.exit(0 if main(NL) else 1)
