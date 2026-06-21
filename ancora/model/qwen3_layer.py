"""
ancora/model/qwen3_layer.py — Qwen3-0.6B decoder layer, wired from ancora/kernels.

Dataflow (one decoder block, pre-norm + residual):

  ── Attention ──────────────────────────────────────────────────────────────
  h          = RMSNorm(x, input_layernorm)                       norm.py
  q,k,v      = h @ {q,k,v}_proj                                  loss._gemm (BF16)
  q,k        = RMSNorm(q), RMSNorm(k)   per-head over head_dim   norm.py  (QK-Norm)
  q,k        = RoPE(q), RoPE(k)                                  rope.py
  o          = FlashAttention(q,k,v)    causal, GQA              attention.py
  x          = x + o @ o_proj                                    loss._gemm + residual
  ── MLP (SwiGLU) ───────────────────────────────────────────────────────────
  h          = RMSNorm(x, post_attention_layernorm)             norm.py
  g,u        = h @ gate_proj, h @ up_proj                        loss._gemm
  a          = silu(g) * u                                       activation.py
  x          = x + a @ down_proj                                 loss._gemm + residual

This is a CORRECTNESS-first assembly: it calls each real kernel and uses host
numpy only for the head reshape/transpose and residual adds (which the megakernel
will fuse on-device later). Activations round-trip device<->host between kernels —
the perf path keeps them resident; here we validate the WIRING (RoPE/QK-Norm
placement, GQA head mapping, residuals, SwiGLU) against an fp64 reference.

head_dim NOTE: head_dim=128 (real Qwen3-0.6B). Attention is BQ=BKV=64, D=128 (gau-nernst
64×64×128 = 94% SOL on sm_120). The framework is hardcoded to this single model target.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataclasses import dataclass

import numpy as np
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # sets CUDA_PATH

from ancora.kernels.loss import _gemm, GTM, GTN, GTK, _GpuArray, f32_to_bf16_bits as _f32bf
from ancora.kernels.norm import rmsnorm_forward, rmsnorm_backward
from ancora.kernels.rope import rope_forward, rope_backward
from ancora.kernels.attention import flash_attn_forward, flash_attn_backward
from ancora.kernels.activation import swiglu_forward, swiglu_backward


def _bf(x):
    """Round an fp32 array to BF16 value (through the bit pattern) — the precision
    every activation/weight is stored at, so kernel and reference agree on inputs."""
    u = x.astype(np.float32).view(np.uint32); u = u + 0x7FFF + ((u >> 16) & 1)
    return ((u >> 16).astype(np.uint32) << 16).view(np.float32)


def linear_bf16(x_f32: np.ndarray, W_f32: np.ndarray, stream_int: int) -> np.ndarray:
    """y = x @ W.  x:(M,K) f32, W:(K,N) f32 (pre-transposed, BF16-valued) → y:(M,N) f32.
    BF16 inputs, fp32 accumulate (loss._gemm). M%128, N%128, K%64 required."""
    M, K = x_f32.shape
    K2, N = W_f32.shape
    assert K == K2 and M % GTM == 0 and N % GTN == 0 and K % GTK == 0, (M, K, N)
    gx = _GpuArray(_f32bf(x_f32)); gw = _GpuArray(_f32bf(W_f32))
    gc = _GpuArray.zeros((M, N), np.float32)
    ct.launch(stream_int, (M // GTM, N // GTN, 1), _gemm, (gx, gw, gc, K // GTK, GTM, GTN, GTK))
    cudart.cudaStreamSynchronize(stream_int)
    y = gc.to_numpy()
    for g in (gx, gw, gc): g.free()
    return y


def linear_bf16_backward(dy, x, W, stream_int):
    """For y = x @ W (x:(M,K), W:(K,N)):  dx = dy @ Wᵀ,  dW = xᵀ @ dy.
    dy:(M,N), x:(M,K), W:(K,N) → dx:(M,K), dW:(K,N). Reuses the BF16 _gemm via host
    transposes of W and x (same trick loss.linear_ce uses). The megakernel keeps
    these on-device; here correctness-first."""
    dx = linear_bf16(dy, np.ascontiguousarray(W.T), stream_int)   # (M,N)@(N,K) → (M,K)
    dW = linear_bf16(np.ascontiguousarray(x.T), dy, stream_int)   # (K,M)@(M,N) → (K,N)
    return dx, dW


@dataclass
class Qwen3Config:
    hidden: int = 1024
    n_heads: int = 16          # Hq
    n_kv_heads: int = 8        # Hkv  (GQA group G = Hq // Hkv = 2)
    head_dim: int = 128        # Qwen3-0.6B real head_dim (attention D=128, hardcoded)
    intermediate: int = 3072
    eps: float = 1e-6
    rope_theta: float = 1e6


class TransformerLayer:
    """One Qwen3 decoder block. Weights are BF16-valued fp32 arrays, stored
    pre-transposed (in, out) for loss._gemm. Exposed via .w for reference checks."""

    def __init__(self, cfg: Qwen3Config, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        H, Hq, Hkv, Dh, I = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
        qd, kd = Hq * Dh, Hkv * Dh

        def W(shape, s=0.02):  # projection weight (in, out), BF16-valued
            return _bf((rng.standard_normal(shape) * s).astype(np.float32))

        def G(n):              # RMSNorm gain ~ 1.0
            return _bf((1.0 + rng.standard_normal(n) * 0.05).astype(np.float32))

        self.w = {
            "input_ln":  G(H),
            "q_proj":    W((H, qd)),  "k_proj": W((H, kd)),  "v_proj": W((H, kd)),
            "q_norm":    G(Dh),       "k_norm": G(Dh),
            "o_proj":    W((qd, H)),
            "post_ln":   G(H),
            "gate_proj": W((H, I)),   "up_proj": W((H, I)),  "down_proj": W((I, H)),
        }

    def forward(self, x: np.ndarray, stream_int: int, return_cache: bool = False):
        """x: (B, S, hidden) f32 → (B, S, hidden) f32. If return_cache, also returns a
        dict of intermediates the backward needs."""
        cfg = self.cfg; w = self.w
        B, S, H = x.shape
        Hq, Hkv, Dh, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
        M = B * S
        xt = x.reshape(M, H).astype(np.float32)
        c = {"B": B, "S": S, "x": xt}

        # ── Attention block ──────────────────────────────────────────────────
        residual = xt
        h, r1 = rmsnorm_forward(xt, w["input_ln"], stream_int, cfg.eps)       # (M, H)
        q = linear_bf16(h, w["q_proj"], stream_int)                          # (M, Hq*Dh)
        k = linear_bf16(h, w["k_proj"], stream_int)                          # (M, Hkv*Dh)
        v = linear_bf16(h, w["v_proj"], stream_int)                          # (M, Hkv*Dh)
        c.update(rstd1=r1, h1=h, q=q, k=k)                                   # q,k pre-QK-Norm

        # QK-Norm: RMSNorm over head_dim, per (token, head)
        qn, rq = rmsnorm_forward(q.reshape(M * Hq,  Dh), w["q_norm"], stream_int, cfg.eps)
        kn, rk = rmsnorm_forward(k.reshape(M * Hkv, Dh), w["k_norm"], stream_int, cfg.eps)
        c.update(rstd_q=rq, rstd_k=rk)

        # token-major (B,S,H,Dh) → head-major (B,H,S,Dh) for RoPE + attention
        qh = qn.reshape(B, S, Hq,  Dh).transpose(0, 2, 1, 3)
        kh = kn.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
        vh = v.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)

        qr = rope_forward(qh, stream_int, cfg.rope_theta)
        kr = rope_forward(kh, stream_int, cfg.rope_theta)

        o, L = flash_attn_forward(qr, kr, vh, stream_int, causal=True, return_lse=True)
        c.update(qr=qr, kr=kr, vh=vh, o_attn=o, L=L)                         # attn-bwd inputs

        o = o.transpose(0, 2, 1, 3).reshape(M, Hq * Dh)                       # back to token-major
        attn = linear_bf16(o, w["o_proj"], stream_int)                       # (M, H)
        c["o_tok"] = o
        xt = _bf(residual + attn)

        # ── MLP (SwiGLU) ─────────────────────────────────────────────────────
        residual = xt
        h, r2 = rmsnorm_forward(xt, w["post_ln"], stream_int, cfg.eps)
        gate = linear_bf16(h, w["gate_proj"], stream_int)                    # (M, I)
        up   = linear_bf16(h, w["up_proj"],   stream_int)                    # (M, I)
        act  = swiglu_forward(gate, up, stream_int)                          # (M, I)
        mlp  = linear_bf16(act, w["down_proj"], stream_int)                  # (M, H)
        c.update(x2=xt, rstd2=r2, h2=h, gate=gate, up=up, act=act)
        xt = _bf(residual + mlp)

        out = xt.reshape(B, S, H)
        return (out, c) if return_cache else out

    def backward(self, d_out: np.ndarray, cache: dict, stream_int: int):
        """d_out: (B,S,hidden) dL/dout. Returns (d_x (B,S,hidden), grads dict).
        Reverse of forward; gradient splits at each residual (add → copy to both
        branches), weight grads accumulated per projection / norm."""
        cfg = self.cfg; w = self.w; c = cache
        Hq, Hkv, Dh, I, H = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.hidden
        B, S = c["B"], c["S"]; M = B * S
        g = {}
        d = d_out.reshape(M, H).astype(np.float32)

        # ── MLP backward ─────────────────────────────────────────────────────
        d_res2 = d                                          # out = x2 + mlp → residual branch
        d_a, g["down_proj"] = linear_bf16_backward(d, c["act"], w["down_proj"], stream_int)
        d_gate, d_up = swiglu_backward(c["gate"], c["up"], d_a, stream_int)
        d_h2a, g["gate_proj"] = linear_bf16_backward(d_gate, c["h2"], w["gate_proj"], stream_int)
        d_h2b, g["up_proj"]   = linear_bf16_backward(d_up,   c["h2"], w["up_proj"],   stream_int)
        d_h2 = d_h2a + d_h2b
        d_x2_mlp, g["post_ln"] = rmsnorm_backward(c["x2"], w["post_ln"], d_h2, c["rstd2"], stream_int)
        d_x2 = d_res2 + d_x2_mlp                            # join residual + through-norm

        # ── Attention backward ───────────────────────────────────────────────
        d_res1 = d_x2                                       # x2 = x + attn → residual branch
        d_o_tok, g["o_proj"] = linear_bf16_backward(d_x2, c["o_tok"], w["o_proj"], stream_int)
        # token-major (M, Hq*Dh) → head-major (B,Hq,S,Dh) for attention backward
        d_o_hm = d_o_tok.reshape(B, S, Hq, Dh).transpose(0, 2, 1, 3)
        d_qr, d_kr, d_vh = flash_attn_backward(c["qr"], c["kr"], c["vh"], c["o_attn"],
                                               d_o_hm, c["L"], stream_int)
        d_qn_hm = rope_backward(d_qr, stream_int, cfg.rope_theta)            # undo RoPE
        d_kn_hm = rope_backward(d_kr, stream_int, cfg.rope_theta)
        # head-major (B,H,S,Dh) → per-(token,head) rows (M*H, Dh) for QK-Norm backward
        d_qn = d_qn_hm.transpose(0, 2, 1, 3).reshape(M * Hq,  Dh)
        d_kn = d_kn_hm.transpose(0, 2, 1, 3).reshape(M * Hkv, Dh)
        d_v  = d_vh.transpose(0, 2, 1, 3).reshape(M, Hkv * Dh)
        d_q, g["q_norm"] = rmsnorm_backward(c["q"].reshape(M * Hq,  Dh), w["q_norm"], d_qn, c["rstd_q"], stream_int)
        d_k, g["k_norm"] = rmsnorm_backward(c["k"].reshape(M * Hkv, Dh), w["k_norm"], d_kn, c["rstd_k"], stream_int)
        d_q = d_q.reshape(M, Hq * Dh); d_k = d_k.reshape(M, Hkv * Dh)
        d_h1q, g["q_proj"] = linear_bf16_backward(d_q, c["h1"], w["q_proj"], stream_int)
        d_h1k, g["k_proj"] = linear_bf16_backward(d_k, c["h1"], w["k_proj"], stream_int)
        d_h1v, g["v_proj"] = linear_bf16_backward(d_v, c["h1"], w["v_proj"], stream_int)
        d_h1 = d_h1q + d_h1k + d_h1v
        d_x_attn, g["input_ln"] = rmsnorm_backward(c["x"], w["input_ln"], d_h1, c["rstd1"], stream_int)
        d_x = d_res1 + d_x_attn                             # join residual + through-norm

        return d_x.reshape(B, S, H), g
