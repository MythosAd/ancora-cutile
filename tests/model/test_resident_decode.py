"""ResidentDecodeLayer — device-resident single-token rollout decode, BITWISE vs training prefill.

The on-policy RL requirement (ratio π_train/π_infer = 1 EXACTLY) reduces to: the decode (rollout)
forward must produce a hidden state for position t that is BITWISE-identical to the training prefill
forward at position t. Then the (shared) boundary gives an identical logprob → ratio = 1, no
importance sampling needed (the single-codebase advantage, [[batch-invariance]]).

This test runs the SAME weights through two forwards:
  • PREFILL (training): a stack of ResidentLayerTrain over the whole sequence at once → hidden (S,H).
  • DECODE  (rollout):  a stack of ResidentDecodeLayer, ONE token at a time, autoregressive, each layer
                        appending its K/V to a per-layer cache (teacher-forced with the same tokens).
and checks:
  (A) decode hidden[t]  ==  prefill hidden[t]   bitwise, every position  (the foundation)
  (B) rollout logprob   ==  training prefill logprob  bitwise  (final norm + tied-head logits + CE)
  (C) decode is deterministic across runs (persistent buffers, no alloc churn).

Run:  python tests/model/test_resident_decode.py [n_layers]
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident import _DBuf, _f32bf
from ancora.model.resident_train import ResidentLayerTrain
from ancora.model.resident_decode import ResidentDecodeLayer, MGEMM
from ancora.kernels.norm import rmsnorm_forward
from ancora.kernels.fused import _gemm_nt_f32
from ancora.kernels.loss import _ce_stats, CTM, TV

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config(); H = cfg.hidden
bitsf = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32)


def _free_obj(objs):
    """Free every device buffer reachable from these objects (dedup by ptr — _DBuf.view shares ptr)."""
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


def prefill(layers_w, x0_all, S):
    """Training prefill: stack of ResidentLayerTrain over the full sequence at once → hidden (S,H) f32."""
    layers = [ResidentLayerTrain(cfg, w, 1, S, sr_grad=False) for w in layers_w]
    gin0 = _DBuf(np.ascontiguousarray(x0_all, np.float32)); gx = gin0
    for l in layers: gx = l.forward(gx, si)
    cudart.cudaStreamSynchronize(si)
    out = gx.to_numpy().copy()
    gin0.free(); _free_obj(layers)
    return out


def decode_rollout(layers_w, x0_all, S, maxS, runs=1):
    """Rollout decode: stack of ResidentDecodeLayer, one token at a time, per-layer KV-cache appended
    each step. Returns the assembled hidden (S,H) f32 (last `runs` runs reuse the SAME layers/cache →
    also a determinism check). x0_all[t] is the layer-0 input (embed[ids[t]]) of step t."""
    Md = MGEMM
    layers = [ResidentDecodeLayer(cfg, w, 1, maxS) for w in layers_w]
    gin = _DBuf.zeros((Md, H), np.float32)               # frontier token at row 0, rows 1.. = padding (0)
    dhid = _DBuf.zeros((S, H), np.float32)
    x0_rows = [np.ascontiguousarray(x0_all[t], np.float32) for t in range(S)]   # persistent async-copy sources
    outs = []
    for _ in range(runs):
        for t in range(S):
            cdrv.cuMemcpyHtoDAsync(gin.ptr, x0_rows[t], H * 4, si)               # upload frontier embed → row 0
            x = gin
            for l in layers: x = l.forward(x, t, si)                            # device-resident layer chain
            cdrv.cuMemcpyDtoDAsync(int(dhid.ptr) + t * H * 4, int(x.ptr), H * 4, si)   # collect hidden[t] = gout row 0
        cudart.cudaStreamSynchronize(si)
        outs.append(dhid.to_numpy().copy())
    gin.free(); dhid.free(); _free_obj(layers)
    return outs


def boundary_logprob(hidden_pre, embed, final_norm, labels, eps, S):
    """Final RMSNorm + tied-head logits + CE → per-token logprob (S,1) f32. Applied IDENTICALLY to the
    prefill and decode hiddens (so identical hiddens ⇒ identical logprobs ⇒ ratio = 1)."""
    V = embed.shape[0]
    h, _ = rmsnorm_forward(hidden_pre, final_norm, si, eps)                      # (S,H) f32
    gh = _DBuf(_f32bf(h)); gembed = _DBuf(_f32bf(embed)); glog = _DBuf.zeros((S, V), np.float32)
    ct.launch(si, (S // 128, V // 128, 1), _gemm_nt_f32, (gh, gembed, glog, H // 64, 128, 128, 64))
    glab = _DBuf(np.ascontiguousarray(labels.astype(np.int32).reshape(S, 1)))
    glp = _DBuf.zeros((S, 1), np.float32); glse = _DBuf.zeros((S, 1), np.float32)
    ct.launch(si, (S // CTM, 1, 1), _ce_stats, (glog, glab, glp, glse, V // TV))
    cudart.cudaStreamSynchronize(si)
    lp = glp.to_numpy().copy()
    for g in (gh, gembed, glog, glab, glp, glse): g.free()
    return lp


def main(NL=2, S=128):
    maxS, V = S, 2048
    print(f"ResidentDecodeLayer — rollout decode vs training prefill   NL={NL} S={S} V={V}")
    print("=" * 76)
    rng = np.random.default_rng(0)
    layers_w = [TransformerLayer(cfg, seed=i).w for i in range(NL)]
    embed = (rng.standard_normal((V, H)) * 0.02).astype(np.float32)
    final_norm = (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)
    ids = rng.integers(0, V, S).astype(np.int64)
    labels = rng.integers(0, V, S).astype(np.int64)
    # layer-0 input = embed[ids] (bf16-valued, the rollout's actual residual entering the stack)
    eb = _f32bf(embed)
    x0_all = (eb[ids].astype(np.uint32) << 16).view(np.float32).astype(np.float32)   # (S,H) f32

    t0 = time.time(); pre = prefill(layers_w, x0_all, S); print(f"  prefill done ({time.time()-t0:.1f}s)")
    t0 = time.time(); dec_runs = decode_rollout(layers_w, x0_all, S, maxS, runs=2); print(f"  decode  done ({time.time()-t0:.1f}s)")
    dec = dec_runs[0]

    # ── (A) hidden bitwise ──
    print("-" * 76)
    same_rows = (bitsf(pre) == bitsf(dec)).all(axis=1)
    nbad = int((~same_rows).sum())
    worst = np.abs(pre - dec).max()
    print(f"  (A) hidden bitwise: {S - nbad}/{S} positions identical, max|Δ|={worst:.3g}  "
          f"{'OK' if nbad == 0 else 'FAIL'}")
    if nbad:
        bad = np.where(~same_rows)[0]
        for t in bad[:6]:
            print(f"        pos {t:3d}: max|Δ|={np.abs(pre[t]-dec[t]).max():.3g}")

    # ── (C) determinism (decode twice, reusing layers/cache) ──
    detok = np.array_equal(bitsf(dec_runs[0]), bitsf(dec_runs[1]))
    print(f"  (C) decode determinism (run×2): {'bitwise IDENTICAL' if detok else 'DIFFER'}  {'OK' if detok else 'FAIL'}")

    # ── (B) end-to-end logprob bitwise ──
    lp_pre = boundary_logprob(pre, embed, final_norm, labels, cfg.eps, S)
    lp_dec = boundary_logprob(dec, embed, final_norm, labels, cfg.eps, S)
    lp_ok = np.array_equal(bitsf(lp_pre), bitsf(lp_dec))
    print(f"  (B) logprob bitwise: rollout == prefill   max|Δ|={np.abs(lp_pre-lp_dec).max():.3g}  "
          f"mean CE={float(-lp_pre.mean()):.4f}  {'OK' if lp_ok else 'FAIL'}")

    ok = (nbad == 0) and detok and lp_ok
    print("=" * 76)
    print(f"  {'PASS — rollout decode is BITWISE-equal to training prefill (ratio=1 exactly)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    sys.exit(0 if main(NL, S) else 1)
