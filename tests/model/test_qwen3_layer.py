"""Qwen3 decoder layer assembly — full forward vs an fp64 numpy reference (validates
the WIRING: RMSNorm, QKV proj, QK-Norm, RoPE, causal GQA attention, o_proj, residual,
SwiGLU MLP), plus end-to-end batch invariance (batch-0 output bitwise identical for
B=1 vs B=2 — the RL train/rollout consistency property, composed across all kernels).

Keep this: re-run after any kernel or toolkit change."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config, _bf
from ancora.kernels.rope import build_cos_sin
from ancora.kernels.norm import f32_to_bf16_bits as f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def rope_ref(x, base):
    B, Hh, S, Dh = x.shape
    c, s = build_cos_sin(S, Dh, base)
    cosf = np.concatenate([c, c], -1)[None, None].astype(np.float64)
    sinf = np.concatenate([s, s], -1)[None, None].astype(np.float64)
    d = Dh // 2
    rot = np.concatenate([-x[..., d:], x[..., :d]], -1)
    return x * cosf + rot * sinf


def ref_forward(x, w, cfg):
    """Pure fp64 reference (same BF16-valued weights; no intermediate rounding)."""
    B, S, H = x.shape
    Hq, Hkv, Dh, G = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
    M = B * S; eps = cfg.eps
    xt = x.reshape(M, H).astype(np.float64)

    def rms(z, g):
        r = 1.0 / np.sqrt((z * z).mean(-1, keepdims=True) + eps)
        return z * r * g.astype(np.float64)

    residual = xt
    h = rms(xt, w["input_ln"])
    q = h @ w["q_proj"].astype(np.float64)
    k = h @ w["k_proj"].astype(np.float64)
    v = h @ w["v_proj"].astype(np.float64)
    q = rms(q.reshape(M * Hq,  Dh), w["q_norm"]).reshape(B, S, Hq,  Dh).transpose(0, 2, 1, 3)
    k = rms(k.reshape(M * Hkv, Dh), w["k_norm"]).reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    q = rope_ref(q, cfg.rope_theta); k = rope_ref(k, cfg.rope_theta)
    k = np.repeat(k, G, axis=1); v = np.repeat(v, G, axis=1)          # GQA expand
    scale = 1.0 / math.sqrt(Dh)
    sc = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
    sc = np.where(np.triu(np.ones((S, S), bool), 1)[None, None], -np.inf, sc)
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc); p = p / p.sum(-1, keepdims=True)
    o = np.einsum('bhqk,bhkd->bhqd', p, v).transpose(0, 2, 1, 3).reshape(M, Hq * Dh)
    xt = residual + o @ w["o_proj"].astype(np.float64)

    residual = xt
    h = rms(xt, w["post_ln"])
    g = h @ w["gate_proj"].astype(np.float64)
    u = h @ w["up_proj"].astype(np.float64)
    a = (g / (1.0 + np.exp(-g))) * u
    xt = residual + a @ w["down_proj"].astype(np.float64)
    return xt.reshape(B, S, H)


def rel(a, b): return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def test_correctness():
    print("--- forward vs fp64 reference ---")
    cfg = Qwen3Config(); layer = TransformerLayer(cfg, seed=0); ok = True
    for (B, S) in [(1, 128), (2, 128)]:
        rng = np.random.default_rng(100 + B)
        x = _bf((rng.standard_normal((B, S, cfg.hidden)) * 0.5).astype(np.float32))
        y = layer.forward(x, si)
        y_r = ref_forward(x, layer.w, cfg)
        e = rel(y, y_r); o = e < 0.05; ok &= o
        print(f"  B={B} S={S}: rel={e*100:.2f}%  {'OK' if o else 'FAIL'}")
    return ok


def test_batch_invariance():
    """Batch-0 sequence is identical whether batched with another seq or alone →
    output bits must match (composed batch invariance of every kernel)."""
    print("--- batch invariance (batch-0 output bitwise identical) ---")
    cfg = Qwen3Config(); layer = TransformerLayer(cfg, seed=1)
    S = 128; rng = np.random.default_rng(7)
    seq0 = _bf((rng.standard_normal((1, S, cfg.hidden)) * 0.5).astype(np.float32))

    y1 = layer.forward(seq0, si)                                           # B=1
    seq1 = _bf((rng.standard_normal((1, S, cfg.hidden)) * 0.5).astype(np.float32))
    x2 = np.concatenate([seq0, seq1], axis=0)                             # B=2, batch0=seq0
    y2 = layer.forward(x2, si)

    same = np.array_equal(f32bf(y1[0]), f32bf(y2[0]))
    print(f"  B=1 vs B=2 batch-0: bits identical={same}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    cfg = Qwen3Config()
    print(f"Qwen3 layer — hidden={cfg.hidden} Hq={cfg.n_heads} Hkv={cfg.n_kv_heads} "
          f"Dh={cfg.head_dim} I={cfg.intermediate}")
    print("=" * 64)
    ok = test_correctness()
    bi = test_batch_invariance()
    print("=" * 64)
    print(f"  correctness {'PASS' if ok else 'FAIL'} | batch-invariance {'PASS' if bi else 'FAIL'}")
