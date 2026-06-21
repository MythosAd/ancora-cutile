"""Full MoEModel end-to-end: overfit a fixed batch → CE collapses. Proves the whole interleaved
dense/MoE + local/global stack (embed → scheduled layers → final norm → LM head) trains: forward,
CE loss, and the full reverse-mode backward through every layer + the embedding scatter-add."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, layer_schedule
from ancora.model.moe_model import MoEModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def _overfit(grouped, tie, lr=0.1, nstep=30):
    """Returns (ce0, min_ce, final_ce). min_ce is the fairness measure: it proves the batch is
    FITTABLE (the backward is correct). Plain SGD overshoots/bounces (esp. tied weights, which
    couple input+output) — real runs use AdamW/Muon; here we just need the loss to reach a fit."""
    cfg = MoEConfig(vocab=512, n_layers=4)               # 4 layers (2 dense + 2 MoE), V=512
    model = MoEModel(cfg, seed=0, grouped=grouped, tie=tie)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab, (2, 64)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (128,)).astype(np.int64)
    ce0 = mn = None; ce = None
    for step in range(nstep):
        h, cache = model.forward(ids, si)
        ce, grads = model.loss_backward(h, labels, cache, si)
        if ce0 is None: ce0 = ce
        mn = ce if mn is None else min(mn, ce)
        model.sgd_step(grads, lr=lr)
    return ce0, mn, ce


def test_overfit_grouped_tied():
    print("--- MoEModel (grouped MoE + tied embed) overfit, lr=0.03 ---")
    print("schedule (is_global, ffn_dense):", layer_schedule(MoEConfig(n_layers=4)))
    ce0, mn, ce = _overfit(grouped=True, tie=True, lr=0.03, nstep=40)
    ok = mn / ce0 < 0.2                                   # reached a fit (min CE < 20% of initial)
    print(f"  CE {ce0:.3f} → min {mn:.4f} ({mn/ce0*100:.1f}% of initial)  {'OK' if ok else 'FAIL'}")
    return ok


def test_overfit_grouped_untied():
    print("--- MoEModel (grouped MoE + untied) overfit, lr=0.1 (stable control) ---")
    ce0, mn, ce = _overfit(grouped=True, tie=False, lr=0.1, nstep=30)
    ok = ce / ce0 < 0.05                                  # untied is stable → final CE collapses hard
    print(f"  CE {ce0:.3f} → {ce:.4f} ({ce/ce0*100:.1f}% of initial)  {'OK' if ok else 'FAIL'}")
    return ok


def test_tie_saves_params():
    """Tied embed/LM-head: one V·H weight instead of two."""
    cfg = MoEConfig(vocab=512, n_layers=4); H, V = cfg.hidden, cfg.vocab
    tied = MoEModel(cfg, grouped=False, tie=True)
    untied = MoEModel(cfg, grouped=False, tie=False)
    has_head_t = hasattr(tied, "lm_head"); has_head_u = hasattr(untied, "lm_head")
    ok = (not has_head_t) and has_head_u
    print(f"--- tie saves V·H = {V*H/1e6:.2f}M params: tied has lm_head={has_head_t}, untied={has_head_u}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 64)
    r = [test_overfit_grouped_untied(), test_overfit_grouped_tied(), test_tie_saves_params()]
    print("=" * 64)
    print("  ALL PASS" if all(r) else "  FAIL")
