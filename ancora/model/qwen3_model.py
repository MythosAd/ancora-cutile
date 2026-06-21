"""
ancora/model/qwen3_model.py — full Qwen3-0.6B: embed + N decoder layers + final
RMSNorm + LM head, wired from the validated TransformerLayer + kernels.

  ids (B,S) ─embed lookup→ x (B,S,H) ─[N × TransformerLayer]→ x ─final RMSNorm→ h
  h ─lm_head + CE (linear_ce)→ loss ;  backward chains every layer's bwd in reverse,
  and the embedding gradient is a scatter-add of d_x at the input token rows.

v1 simplifications (noted for faithfulness to real Qwen3-0.6B):
  • UNTIED embed / lm_head. Qwen3-0.6B sets tie_word_embeddings=True → lm_head = embedᵀ,
    and d_embed = scatter_add(lookup) + dW_headᵀ. TODO: tie (halves the V·H params).
  • head_dim=64 (attention kernel limit; real = 128).
Default n_layers small for tests; real config = 28 layers, V=151936.

Param routing for the HybridOptimizer (by name):
  Muon  → 'layerI.{q,k,v,o,gate,up,down}_proj'  (2D matmul weights)
  AdamW → 'embed', 'lm_head', 'final_norm', 'layerI.{input_ln,post_ln,q_norm,k_norm}'
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import ancora.env  # sets CUDA_PATH

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.loss import linear_ce


class Qwen3Model:
    def __init__(self, cfg: Qwen3Config, n_layers: int, vocab: int, seed: int = 0):
        self.cfg, self.V, self.nl = cfg, vocab, n_layers
        H = cfg.hidden
        rng = np.random.default_rng(seed)
        self.embed = _bf((rng.standard_normal((vocab, H)) * 0.02).astype(np.float32))
        self.layers = [TransformerLayer(cfg, seed=seed + 1 + i) for i in range(n_layers)]
        self.final_norm = _bf((1.0 + rng.standard_normal(H) * 0.05).astype(np.float32))
        self.lm_head = _bf((rng.standard_normal((H, vocab)) * 0.02).astype(np.float32))

    # ── param plumbing for the optimizer ─────────────────────────────────────
    def params(self):
        p = {"embed": self.embed, "lm_head": self.lm_head, "final_norm": self.final_norm}
        for i, l in enumerate(self.layers):
            for n, w in l.w.items():
                p[f"layer{i}.{n}"] = w
        return p

    def load(self, weights):
        self.embed = weights["embed"]; self.lm_head = weights["lm_head"]
        self.final_norm = weights["final_norm"]
        for i, l in enumerate(self.layers):
            for n in l.w:
                l.w[n] = weights[f"layer{i}.{n}"]

    # ── forward / loss+backward ──────────────────────────────────────────────
    def forward(self, input_ids: np.ndarray, stream_int: int):
        """input_ids (B,S) int → (hidden (M,H) f32 after final norm, cache)."""
        B, S = input_ids.shape; H = self.cfg.hidden; M = B * S
        x = self.embed[input_ids.reshape(-1)].reshape(B, S, H).astype(np.float32)  # gather
        caches = []
        for l in self.layers:
            x, c = l.forward(x, stream_int, return_cache=True)
            caches.append(c)
        h, rstd_f = rmsnorm_forward(x.reshape(M, H), self.final_norm, stream_int, self.cfg.eps)
        cache = dict(caches=caches, x_pre=x.reshape(M, H), rstd_f=rstd_f, ids=input_ids)
        return h, cache

    def loss_backward(self, hidden, labels, cache, stream_int, advantage=None):
        """hidden (M,H), labels (M,). Returns (ce_loss, grads dict matching params()).
        advantage None → SFT (L=-mean logprob); else GRPO weighting."""
        M, H = hidden.shape
        adv = np.ones(M, np.float32) if advantage is None else advantage.astype(np.float32)
        lp, dhidden, dW_head = linear_ce(hidden, self.lm_head, labels, stream_int, advantage=adv)
        ce = float(-(adv * lp).mean())

        d_xpre, d_final = rmsnorm_backward(cache["x_pre"], self.final_norm, dhidden,
                                           cache["rstd_f"], stream_int)
        grads = {"lm_head": dW_head, "final_norm": d_final}

        B, S = cache["ids"].shape
        d = d_xpre.reshape(B, S, H)
        for i in reversed(range(self.nl)):
            d, lg = self.layers[i].backward(d, cache["caches"][i], stream_int)
            for n, g in lg.items():
                grads[f"layer{i}.{n}"] = g

        d_embed = np.zeros_like(self.embed)
        np.add.at(d_embed, cache["ids"].reshape(-1), d.reshape(M, H))   # scatter-add
        grads["embed"] = d_embed
        return ce, grads
