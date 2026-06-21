"""
ancora/optim/hybrid.py — Muon + AdamW hybrid (the modern LLM setup).

Routing (standard, from the Muon paper / nanoGPT speedrun / Kimi):
  • Muon  → 2D hidden weight MATRICES (q/k/v/o/gate/up/down_proj).
  • AdamW → everything else: 1D RMSNorm gains, embedding, LM-head, biases.
Muon on embeddings/heads HURTS — the routing is part of the recipe, not an optimization.

Usage:
  opt = HybridOptimizer(params, muon_matched=lambda n,p: p.ndim==2 and "norm" not in n,
                        no_decay=lambda n,p: p.ndim==1)
  opt.step(grads, stream); w = opt.weights()   # BF16-valued weights for the next forward
"""
from .muon import Muon
from .adamw import AdamW


def default_is_muon(name, p):
    """2D matrix that isn't a norm gain / embedding / lm_head → Muon."""
    if p.ndim != 2:
        return False
    low = name.lower()
    if any(t in low for t in ("norm", "ln", "embed", "lm_head", "head", "wte", "wpe")):
        return False
    return True


class HybridOptimizer:
    def __init__(self, params, is_muon=default_is_muon, no_decay=None,
                 muon_kw=None, adamw_kw=None, si=None, muon_device=False):
        muon_kw = dict(lr=0.02, momentum=0.95, ns_steps=5, **(muon_kw or {}))
        if muon_device:                                 # run Newton-Schulz matmuls on the GPU
            muon_kw.update(device=True, si=si)
        adamw_kw = adamw_kw or dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.01)
        if no_decay is None:
            no_decay = lambda n, p: (p.ndim == 1)   # don't decay 1D gains/biases

        self.muon_names = [n for n, p in params.items() if is_muon(n, p)]
        self.adamw_names = [n for n in params if n not in self.muon_names]
        self.muon = Muon({n: params[n] for n in self.muon_names}, **muon_kw)
        self.adamw = AdamW({n: params[n] for n in self.adamw_names},
                           no_decay=tuple(n for n in self.adamw_names if no_decay(n, params[n])),
                           **adamw_kw)

    def step(self, grads, stream_int):
        self.muon.step({n: grads[n] for n in self.muon_names}, stream_int)
        self.adamw.step({n: grads[n] for n in self.adamw_names}, stream_int)

    def weights(self):
        return {**self.muon.weights(), **self.adamw.weights()}

    def routing(self):
        return {"muon": list(self.muon_names), "adamw": list(self.adamw_names)}

    def free(self):
        self.muon.free(); self.adamw.free()
