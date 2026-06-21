"""ROLLOUT-QUALITY GATE for NVFP4: does the ~13% per-GEMM FP4 error COMPOUND across the full
model depth and degrade generation? Builds a faithful numpy Qwen3 forward (28 layers) where ONLY
the 7 projections (+ optionally the LM head) swap BF16↔NVFP4. NVFP4 GEMM is simulated as the
dequant-operand product — which the CUTLASS NVFP4 kernel was proven to match at 0.166%
(test_nvfp4_gemm.py), so this is a faithful accuracy proxy. Reports the per-layer hidden-state
error growth + final-logit top-1 / top-5 token agreement + softmax KL (the rollout-relevant metrics).
NOTE: random (untrained) weights — shows the error-growth STRUCTURE, not a trained model's exact %s."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.kernels.quant_nvfp4 import quantize_nvfp4_rowblock, dequantize_nvfp4
from ancora.kernels.quant import quantize_rowblock, dequantize_rowblock   # MXFP8 control

_bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
bfv = lambda x: (_bf(x).astype(np.uint32) << 16).view(np.float32)   # round f32→bf16
rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)

# ── projection matmuls (the only thing that differs between runs) ──
def mm_bf16(x, w):  return bfv(x).astype(np.float64) @ bfv(w).astype(np.float64)
def _qn_act(x): return dequantize_nvfp4(*quantize_nvfp4_rowblock(x.astype(np.float32)))
def _qn_wt(w):  return dequantize_nvfp4(*quantize_nvfp4_rowblock(np.ascontiguousarray(w.T).astype(np.float32))).T
def mm_nvfp4(x, w): return _qn_act(x).astype(np.float64) @ _qn_wt(w).astype(np.float64)
def _q8_act(x): return dequantize_rowblock(*quantize_rowblock(x.astype(np.float32)))
def _q8_wt(w):  return dequantize_rowblock(*quantize_rowblock(np.ascontiguousarray(w.T).astype(np.float32))).T
def mm_mxfp8(x, w): return _q8_act(x).astype(np.float64) @ _q8_wt(w).astype(np.float64)   # MXFP8 control

def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w).astype(np.float32)

def build_cos_sin(S, D, base):
    inv = base ** (-(np.arange(D // 2) * 2.0 / D)); ang = np.arange(S)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def rope(x, cos, sin):  # x (S,Hh,D)
    D = x.shape[-1]; x1, x2 = x[..., :D // 2], x[..., D // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

def attention(q, k, v, Hq, Hkv, D, scale):  # q (S,Hq,D) k,v (S,Hkv,D), causal GQA, f32
    S = q.shape[0]; G = Hq // Hkv; O = np.zeros((S, Hq, D), np.float32)
    mask = np.tril(np.ones((S, S), bool))
    for h in range(Hq):
        kvh = h // G
        sc = (q[:, h].astype(np.float64) @ k[:, kvh].T.astype(np.float64)) * scale
        sc = np.where(mask, sc, -1e30); sc -= sc.max(1, keepdims=True)
        p = np.exp(sc); p /= p.sum(1, keepdims=True)
        O[:, h] = (p @ v[:, kvh].astype(np.float64)).astype(np.float32)
    return O


def layer_fwd(x, w, cfg, cos, sin, prec):
    """prec: dict projection-name → matmul fn (per-projection precision)."""
    H, Hq, Hkv, Dh, I, eps = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate, cfg.eps
    S = x.shape[0]; scale = 1.0 / np.sqrt(Dh); P = lambda nm, a: prec[nm](a, w[nm])
    h = rmsnorm(x, w["input_ln"], eps)
    q = P("q_proj", h).reshape(S, Hq, Dh); k = P("k_proj", h).reshape(S, Hkv, Dh); v = P("v_proj", h).reshape(S, Hkv, Dh)
    q = rmsnorm(q, w["q_norm"], eps); k = rmsnorm(k, w["k_norm"], eps)
    q = rope(q, cos, sin); k = rope(k, cos, sin)
    o = attention(q, k, v, Hq, Hkv, Dh, scale).reshape(S, Hq * Dh)
    x = x + P("o_proj", o)
    h2 = rmsnorm(x, w["post_ln"], eps)
    g = P("gate_proj", h2); u = P("up_proj", h2)
    act = (g * (1.0 / (1.0 + np.exp(-g)))) * u
    x = x + P("down_proj", act)
    return x.astype(np.float32)

PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# projection FLOP weights (K*N), for the FP4-fraction column
_FW = {"q_proj": 1024*1024, "k_proj": 1024*512, "v_proj": 1024*512, "o_proj": 1024*1024,
       "gate_proj": 1024*3072, "up_proj": 1024*3072, "down_proj": 3072*1024}
def prec_cfg(fp4_set, base=mm_mxfp8):
    return {nm: (mm_nvfp4 if nm in fp4_set else base) for nm in PROJS}
def fp4_frac(fp4_set):
    return sum(_FW[n] for n in fp4_set) / sum(_FW.values())


def agree(lb, lq):   # token-agreement metrics of quantized logits vs bf16 reference
    t1b = lb.argmax(-1); a1 = (t1b == lq.argmax(-1)).mean()
    top5 = np.argsort(lq, -1)[:, -5:]; a5 = np.mean([t1b[i] in top5[i] for i in range(len(t1b))])
    return a1, a5


def run(L=28, S=256, vocab=32000, seed=0):
    cfg = Qwen3Config(); H = cfg.hidden
    rng = np.random.default_rng(seed)
    layers = [TransformerLayer(cfg, seed=i).w for i in range(L)]
    embed = (rng.standard_normal((vocab, H)) * 0.02).astype(np.float32)
    final_norm = (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)
    lm_head = (rng.standard_normal((H, vocab)) * 0.02).astype(np.float32)
    cos, sin = build_cos_sin(S, cfg.head_dim, cfg.rope_theta)
    x0 = embed[rng.integers(0, vocab, S)].astype(np.float32)
    print(f"  depth L={L}  S={S}  vocab={vocab}  (lm_head BF16; non-FP4 projections = MXFP8 baseline)")
    configs = {
        "all BF16  (ref)":  ({nm: mm_bf16 for nm in PROJS}, 0.0),
        "all MXFP8 (base)": (prec_cfg(set()), 0.0),
        "down FP4 +MXFP8":  (prec_cfg({"down_proj"}), fp4_frac({"down_proj"})),
        "FFN  FP4 +MXFP8":  (prec_cfg({"gate_proj", "up_proj", "down_proj"}), fp4_frac({"gate_proj", "up_proj", "down_proj"})),
        "all  FP4":         (prec_cfg(set(PROJS), base=mm_nvfp4), 1.0),
    }
    finals, lastx = {}, {}
    for nm, (prec, _) in configs.items():
        x = x0.copy()
        for i in range(L):
            x = layer_fwd(x, layers[i], cfg, cos, sin, prec)
        lastx[nm] = x; finals[nm] = mm_bf16(rmsnorm(x, final_norm, cfg.eps), lm_head)
    lb = finals["all BF16  (ref)"]; xref = lastx["all BF16  (ref)"]
    print(f"  {'config':17s}{'FP4%FLOP':>9s}{'hiddenRMS':>11s}{'top-1':>9s}{'top-5':>9s}")
    for nm, (_, frac) in configs.items():
        a1, a5 = agree(lb, finals[nm])
        print(f"  {nm:17s}{frac*100:8.0f}%{rms(lastx[nm], xref)*100:10.0f}%{a1*100:8.1f}%{a5*100:8.1f}%")


if __name__ == "__main__":
    print("NVFP4 SELECTIVE-precision rollout-quality (random untrained weights → RELATIVE signal)"); print("=" * 70)
    run(L=28, S=256)
    print("=" * 70)
    print("  'all MXFP8' = the rollout baseline; each FP4 row = INCREMENTAL cost of moving those")
    print("  projections to FP4 (capturing FP4%FLOP of the projections at ~1.7×). If 'down FP4' or")
    print("  'FFN FP4' stays near the MXFP8 row → mixed-precision FP4 rollout is viable.")
