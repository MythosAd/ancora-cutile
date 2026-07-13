"""
Tests for ancora/rl/grpo.py — advantage normalization (decoupled formula),
KL decoupling, and end-to-end with the fused log-prob kernel.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.rl.grpo import grpo_advantage, broadcast_to_tokens, grpo_loss, kl_k3
from ancora.kernels.loss import fused_logprob

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def test_advantage_formulas():
    """Formula is host-side & swappable — verify std/mean/none vs manual."""
    print("--- advantage normalization (decoupled formula) ---")
    rewards = np.array([1., 2., 3., 4.,   0., 0., 0., 4.], np.float32)  # 2 groups, G=4
    ok = True
    # std
    A = grpo_advantage(rewards, 4, norm="std")
    g0 = (np.array([1,2,3,4]) - 2.5) / (np.std([1,2,3,4]) + 1e-4)
    ok &= np.allclose(A[:4], g0, atol=1e-3)
    # mean
    A2 = grpo_advantage(rewards, 4, norm="mean")
    g0m = (np.array([1,2,3,4]) - 2.5) / (abs(2.5) + 1e-4)
    ok &= np.allclose(A2[:4], g0m, atol=1e-3)
    # none
    A3 = grpo_advantage(rewards, 4, norm="none")
    ok &= np.allclose(A3[:4], np.array([1,2,3,4]) - 2.5, atol=1e-3)
    # advantage sums ~0 within a group (centered)
    ok &= abs(A[:4].sum()) < 1e-3
    # DEFAULT is the ML form (== norm="mean"): adv = (r-mean)/(|mean|+eps)
    ok &= np.array_equal(grpo_advantage(rewards, 4), A2)
    print(f"  std/mean/none match manual, default==ML(mean), group-centered (Σ≈0): {'OK' if ok else 'FAIL'}")
    return ok


def test_kl_decoupling():
    """beta=0 → KL skipped, no ref_logprob needed. beta>0 → k3 KL added."""
    print("--- KL decoupling ---")
    rng = np.random.default_rng(0)
    M = 32
    logprob = rng.standard_normal(M).astype(np.float32) - 2
    adv     = rng.standard_normal(M).astype(np.float32)

    # beta=0: works with NO ref model
    loss0, pg0, kl0 = grpo_loss(logprob, adv, ref_logprob=None, beta=0.0)
    ok1 = (kl0 == 0.0) and abs(loss0 - pg0) < 1e-6 and abs(pg0 - (-(adv*logprob).mean())) < 1e-5
    print(f"  beta=0: KL skipped, loss=pg=-mean(adv·logπ), no ref needed: {'OK' if ok1 else 'FAIL'}")

    # beta>0: KL added, equals manual k3
    ref = logprob + 0.1 * rng.standard_normal(M).astype(np.float32)
    loss1, pg1, kl1 = grpo_loss(logprob, adv, ref_logprob=ref, beta=0.5)
    kl_manual = float(kl_k3(logprob, ref).mean())
    ok2 = abs(kl1 - kl_manual) < 1e-5 and abs(loss1 - (pg1 + 0.5*kl1)) < 1e-5 and kl1 >= 0
    print(f"  beta=0.5: loss=pg+β·KL, KL=k3≥0 matches manual: {'OK' if ok2 else 'FAIL'}")
    return ok1 and ok2


def test_end_to_end():
    """fused_logprob (kernel) → grpo_advantage → grpo_loss, full path."""
    print("--- end-to-end (kernel logprob → GRPO loss) ---")
    rng = np.random.default_rng(1)
    G, tokens_per = 4, 16              # 4 completions, 16 tokens each
    M = G * tokens_per                  # 64 tokens total (= TM)
    H, V = 512, 1024
    hidden = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
    w_head = (rng.standard_normal((H, V)) * 0.5).astype(np.float32)
    labels = rng.integers(0, V, M)

    logprob = fused_logprob(hidden, w_head, labels, si)        # (M,) from kernel

    rewards = np.array([0.2, 0.8, 0.5, 1.0], np.float32)        # per completion
    adv_c   = grpo_advantage(rewards, G, norm="std")            # (G,)
    tok_cid = np.repeat(np.arange(G), tokens_per)               # token→completion
    adv_t   = broadcast_to_tokens(adv_c, tok_cid)              # (M,)

    loss, pg, kl = grpo_loss(logprob, adv_t, beta=0.0)
    manual = -float((adv_t * logprob).mean())
    ok = abs(loss - manual) < 1e-4 and np.isfinite(loss)
    print(f"  GRPO loss (no KL)={loss:.4f} matches manual={manual:.4f}: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ancora/rl/grpo.py — advantage + KL (decoupled from kernel)")
    print("=" * 60)
    r = [test_advantage_formulas(), test_kl_decoupling(), test_end_to_end()]
    print("=" * 60)
    print(f"  {sum(r)}/{len(r)} passed"
          + ("  → advantage/KL decoupled, kernel untouched ✓" if all(r) else "  → FAIL"))
