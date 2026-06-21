"""
ancora/model/moe_model.py — full interleaved dense/MoE + local/global model: embed + N
MoEDecoderLayers (per layer_schedule) + final RMSNorm + LM head, wired from moe_layer.py.

  ids (B,S) ─embed→ x ─[N × MoEDecoderLayer (dense|moe FFN, local|global attn)]→ x
  ─final RMSNorm→ h ─lm_head+CE→ loss ;  backward chains every layer in reverse, embed grad
  is a scatter-add of d_x at the input token rows. Mirror of qwen3_model.Qwen3Model.

  tie=True (default, Qwen3 tie_word_embeddings): lm_head = embedᵀ → ONE weight; d_embed =
    scatter_add(input) + dW_headᵀ. grouped=True: MoE layers use kernels/moe.GroupedMoEFFN
    (one launch/stage over all experts, preallocated → deterministic, no alloc-churn).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import ancora.env

from ancora.model.moe_layer import MoEConfig, MoEDecoderLayer, layer_schedule, _bf
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.loss import linear_ce


class MoEModel:
    def __init__(self, cfg: MoEConfig, seed: int = 0, grouped: bool = False, tie: bool = True):
        self.cfg = cfg; self.V = cfg.vocab; self.nl = cfg.n_layers
        self.tie, self.grouped = tie, grouped
        H = cfg.hidden; rng = np.random.default_rng(seed)
        self.embed = _bf((rng.standard_normal((cfg.vocab, H)) * 0.02).astype(np.float32))
        self.sched = layer_schedule(cfg)
        self.layers = [MoEDecoderLayer(cfg, is_global=g, ffn_dense=d, seed=seed + 1 + i, grouped=grouped)
                       for i, (g, d) in enumerate(self.sched)]
        self.final_norm = _bf((1.0 + rng.standard_normal(H) * 0.05).astype(np.float32))
        if not tie:
            self.lm_head = _bf((rng.standard_normal((H, cfg.vocab)) * 0.02).astype(np.float32))

    def _head(self):
        return np.ascontiguousarray(self.embed.T) if self.tie else self.lm_head  # (H,V)

    def forward(self, input_ids, stream_int):
        B, S = input_ids.shape; H = self.cfg.hidden; M = B * S
        x = self.embed[input_ids.reshape(-1)].reshape(B, S, H).astype(np.float32)
        caches = []
        for l in self.layers:
            x, c = l.forward(x, stream_int, return_cache=True)
            caches.append(c)
        h, rstd_f = rmsnorm_forward(x.reshape(M, H), self.final_norm, stream_int, self.cfg.eps)
        return h, dict(caches=caches, x_pre=x.reshape(M, H), rstd_f=rstd_f, ids=input_ids)

    # ── flat param plumbing for the HybridOptimizer ──────────────────────────
    def params(self):
        """Flat {name: weight}. Names: 'embed','final_norm'(,'lm_head'), 'layerI.<attn>',
        'layerI.ffn.<ffn>'. Optimizer routing (default_is_muon): 2D attn/dense-FFN/router → Muon;
        3D MoE-expert weights + embed/head + 1D norms → AdamW."""
        p = {"embed": self.embed, "final_norm": self.final_norm}
        if not self.tie: p["lm_head"] = self.lm_head
        for i, l in enumerate(self.layers):
            for n, w in l.attn.items(): p[f"layer{i}.{n}"] = w
            for n, w in l.ffn.w.items(): p[f"layer{i}.ffn.{n}"] = w
        return p

    def set_weights(self, w):
        """Load BF16-valued weights (from opt.weights()) back into the model."""
        self.embed = w["embed"]; self.final_norm = w["final_norm"]
        if not self.tie: self.lm_head = w["lm_head"]
        for i, l in enumerate(self.layers):
            for n in l.attn: l.attn[n] = w[f"layer{i}.{n}"]
            for n in l.ffn.w: l.ffn.w[n] = w[f"layer{i}.ffn.{n}"]
            if hasattr(l.ffn, "_packed"): l.ffn._packed = False             # grouped MoE: re-pack next fwd

    def loss_backward(self, hidden, labels, cache, stream_int, advantage=None):
        """Returns (ce, grads) with grads FLAT matching params(). advantage None → SFT."""
        M, H = hidden.shape
        adv = np.ones(M, np.float32) if advantage is None else advantage.astype(np.float32)
        lp, dhidden, dW_head = linear_ce(hidden, self._head(), labels, stream_int, advantage=adv)
        ce = float(-(adv * lp).mean())
        d_xpre, d_final = rmsnorm_backward(cache["x_pre"], self.final_norm, dhidden, cache["rstd_f"], stream_int)
        grads = {"final_norm": d_final}

        B, S = cache["ids"].shape; d = d_xpre.reshape(B, S, H)
        for i in reversed(range(self.nl)):
            d, lg = self.layers[i].backward(d, cache["caches"][i], stream_int)
            for n, gg in lg.items():
                if n == "ffn":
                    for fn, fg in gg.items(): grads[f"layer{i}.ffn.{fn}"] = fg
                else:
                    grads[f"layer{i}.{n}"] = gg

        d_embed = np.zeros_like(self.embed)
        np.add.at(d_embed, cache["ids"].reshape(-1), d.reshape(M, H))       # input-embedding grad
        if self.tie:
            d_embed = d_embed + dW_head.T                                   # + tied LM-head grad
        else:
            grads["lm_head"] = dW_head
        grads["embed"] = d_embed
        return ce, grads

    def sgd_step(self, grads, lr):
        """Plain SGD (validation only; real runs use HybridOptimizer)."""
        self.set_weights({n: _bf(p - lr * grads[n]) for n, p in self.params().items()})
