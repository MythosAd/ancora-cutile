"""
Correctness + batch-invariance for ancora/kernels/loss.py (fused log-prob / CE).
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.kernels.loss import fused_logprob, fused_backward, cross_entropy, TM, TV

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def ref_logprob(hidden, w_head, labels):
    """numpy float64 reference: log_softmax(hidden@w_head)[label]."""
    logits = hidden.astype(np.float64) @ w_head.astype(np.float64)   # (M, V)
    m = logits.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
    tgt = logits[np.arange(len(labels)), labels]
    return tgt - lse


def test_correctness():
    print("--- correctness (vs numpy float64) ---")
    rng = np.random.default_rng(0)
    ok_all = True
    for (M, H, V) in [(64, 256, 512), (128, 512, 1024), (256, 1024, 2048)]:
        hidden = rng.standard_normal((M, H)).astype(np.float32) * 0.5
        w_head = rng.standard_normal((H, V)).astype(np.float32) * 0.5
        labels = rng.integers(0, V, M)
        lp  = fused_logprob(hidden, w_head, labels, si)
        ref = ref_logprob(hidden, w_head, labels)
        err = np.abs(lp - ref).max()
        # log-prob magnitudes ~ log(V); relative to that
        rel = err / (np.abs(ref).max() + 1e-9)
        ok = rel < 0.03
        ok_all &= ok
        print(f"  M={M} H={H} V={V}: max_err={err:.4f} rel={rel*100:.2f}%  {'OK' if ok else 'FAIL'}")
    return ok_all


def test_batch_invariance():
    """The first TM rows must be bitwise identical whether run alone or in a big batch."""
    print("--- batch invariance (bitwise) ---")
    rng = np.random.default_rng(1)
    H, V = 512, 1024
    Mbig = 4 * TM
    hidden = rng.standard_normal((Mbig, H)).astype(np.float32) * 0.5
    w_head = rng.standard_normal((H, V)).astype(np.float32) * 0.5
    labels = rng.integers(0, V, Mbig)

    lp_big   = fused_logprob(hidden, w_head, labels, si)
    lp_small = fused_logprob(hidden[:TM], w_head, labels[:TM], si)

    ok = np.array_equal(lp_big[:TM], lp_small)
    diff = np.abs(lp_big[:TM] - lp_small).max()
    print(f"  first {TM} rows: alone vs in batch-of-{Mbig}: "
          f"{'bitwise IDENTICAL' if ok else f'DIFFER maxdiff={diff:.2e}'}  {'OK' if ok else 'FAIL'}")
    return ok


def ref_grads(hidden, w_head, labels, adv):
    """numpy float64 reference gradient of L = -(1/M)Σ adv·logprob."""
    M, V = hidden.shape[0], w_head.shape[1]
    logits = hidden.astype(np.float64) @ w_head.astype(np.float64)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    onehot = np.zeros((M, V)); onehot[np.arange(M), labels] = 1.0
    glogit = (adv[:, None] / M) * (p - onehot)
    return glogit @ w_head.T.astype(np.float64), hidden.T.astype(np.float64) @ glogit


def test_backward():
    print("--- backward gradients (vs numpy float64) ---")
    rng = np.random.default_rng(3)
    ok = True
    for (M, H, V) in [(64, 256, 512), (128, 512, 1024)]:
        hidden = (rng.standard_normal((M, H)) * 0.3).astype(np.float32)
        w_head = (rng.standard_normal((H, V)) * 0.3).astype(np.float32)
        labels = rng.integers(0, V, M)
        adv    = rng.standard_normal(M).astype(np.float32)

        _, lse = fused_logprob(hidden, w_head, labels, si, return_lse=True)
        dh, dw = fused_backward(hidden, w_head, labels, adv, lse, si)
        dh_ref, dw_ref = ref_grads(hidden, w_head, labels, adv)

        rdh = np.abs(dh - dh_ref).max() / (np.abs(dh_ref).max() + 1e-9)
        rdw = np.abs(dw - dw_ref).max() / (np.abs(dw_ref).max() + 1e-9)
        o = rdh < 0.03 and rdw < 0.03; ok &= o
        print(f"  M={M} H={H} V={V}: dhidden rel={rdh*100:.2f}% dwhead rel={rdw*100:.2f}%  "
              f"{'OK' if o else 'FAIL'}")
    return ok


def test_backward_invariance():
    print("--- backward batch invariance (bitwise dhidden) ---")
    rng = np.random.default_rng(4)
    H, V, Mbig = 512, 1024, 4 * TM
    hidden = (rng.standard_normal((Mbig, H)) * 0.3).astype(np.float32)
    w_head = (rng.standard_normal((H, V)) * 0.3).astype(np.float32)
    labels = rng.integers(0, V, Mbig)
    adv    = rng.standard_normal(Mbig).astype(np.float32)

    _, lse_big = fused_logprob(hidden, w_head, labels, si, return_lse=True)
    dh_big, _ = fused_backward(hidden, w_head, labels, adv, lse_big, si)
    # dhidden for first TM rows depends only on those rows (per-m-block) → invariant.
    # Use the SAME inv_M (=1/Mbig) so the scale matches; compare first TM rows.
    _, lse_s = fused_logprob(hidden[:TM], w_head, labels[:TM], si, return_lse=True)
    # match scale: backward divides by its own M, so scale adv by Mbig/TM for the small run
    dh_s, _ = fused_backward(hidden[:TM], w_head, labels[:TM],
                             adv[:TM] * (TM / Mbig), lse_s, si)
    ok = np.array_equal(dh_big[:TM], dh_s)
    diff = np.abs(dh_big[:TM] - dh_s).max()
    print(f"  dhidden first {TM} rows: {'bitwise IDENTICAL' if ok else f'DIFFER max={diff:.2e}'}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def test_cross_entropy():
    print("--- cross-entropy sanity ---")
    rng = np.random.default_rng(2)
    M, H, V = 128, 512, 1024
    hidden = rng.standard_normal((M, H)).astype(np.float32) * 0.5
    w_head = rng.standard_normal((H, V)).astype(np.float32) * 0.5
    labels = rng.integers(0, V, M)
    ce  = cross_entropy(hidden, w_head, labels, si)
    ref = float(-ref_logprob(hidden, w_head, labels).mean())
    ok = abs(ce - ref) / ref < 0.03
    print(f"  CE={ce:.4f} ref={ref:.4f}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"ancora/kernels/loss.py — fused log-prob  TM={TM} TV={TV}")
    print("=" * 60)
    r = [test_correctness(), test_batch_invariance(), test_cross_entropy(),
         test_backward(), test_backward_invariance()]
    print("=" * 60)
    print(f"  {sum(r)}/{len(r)} passed"
          + ("  → fused batch-invariant CE works ✓" if all(r) else "  → FAIL"))
