"""DEFINITIVE FP4 rollout-quality test on the REAL trained Qwen3-0.6B (head_dim=128, vocab=151936).
Resolves the caveat from test_nvfp4_rollout_quality.py (random weights overstated FP4 damage). Same
selective-precision sweep (BF16 / MXFP8 / down-FP4 / FFN-FP4 / all-FP4), real weights, real depth (28).
NVFP4 GEMM ≡ dequant product (proven 0.166%), so the numpy quant→dequant matmul is faithful.
NOTE: random token ids (no tokenizer pkg) — precision comparison on identical input is still valid;
a real prompt would only sharpen it."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from ancora.model.load_qwen3 import load_qwen3
from ancora.kernels.quant_nvfp4 import quantize_nvfp4_rowblock, dequantize_nvfp4
from ancora.kernels.quant import quantize_rowblock, dequantize_rowblock

bfv = lambda x: ((x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint32) << 16).view(np.float32)
def mm_bf16(x, w): return bfv(x).astype(np.float32) @ bfv(w).astype(np.float32)
def _qn_a(x): return dequantize_nvfp4(*quantize_nvfp4_rowblock(x.astype(np.float32)))
def _qn_w(w): return dequantize_nvfp4(*quantize_nvfp4_rowblock(np.ascontiguousarray(w.T).astype(np.float32))).T
def mm_nvfp4(x, w): return _qn_a(x).astype(np.float32) @ _qn_w(w).astype(np.float32)
def _q8_a(x): return dequantize_rowblock(*quantize_rowblock(x.astype(np.float32)))
def _q8_w(w): return dequantize_rowblock(*quantize_rowblock(np.ascontiguousarray(w.T).astype(np.float32))).T
def mm_mxfp8(x, w): return _q8_a(x).astype(np.float32) @ _q8_w(w).astype(np.float32)

H, Hq, Hkv, Dh, I, EPS, THETA = 1024, 16, 8, 128, 3072, 1e-6, 1e6
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_FW = {"q_proj": H * Hq * Dh, "k_proj": H * Hkv * Dh, "v_proj": H * Hkv * Dh, "o_proj": Hq * Dh * H,
       "gate_proj": H * I, "up_proj": H * I, "down_proj": I * H}
prec_cfg = lambda s, base=mm_mxfp8: {n: (mm_nvfp4 if n in s else base) for n in PROJS}
fp4_frac = lambda s: sum(_FW[n] for n in s) / sum(_FW.values())

def rmsnorm(x, w, eps=EPS):
    x = x.astype(np.float32); return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w).astype(np.float32)
def cos_sin(S):
    inv = THETA ** (-(np.arange(Dh // 2) * 2.0 / Dh)); a = np.arange(S)[:, None] * inv[None, :]
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
def rope(x, c, s):
    x1, x2 = x[..., :Dh // 2], x[..., Dh // 2:]; c, s = c[:, None, :], s[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
def attn(q, k, v):
    S = q.shape[0]; G = Hq // Hkv; O = np.zeros((S, Hq, Dh), np.float32); m = np.tril(np.ones((S, S), bool)); sc = 1.0 / np.sqrt(Dh)
    for h in range(Hq):
        s = (q[:, h] @ k[:, h // G].T) * sc; s = np.where(m, s, -1e30); s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True); O[:, h] = p @ v[:, h // G]
    return O

def layer(x, w, c, s, P):
    S = x.shape[0]; mm = lambda nm, a: P[nm](a, w[nm])
    h = rmsnorm(x, w["input_ln"])
    q = mm("q_proj", h).reshape(S, Hq, Dh); k = mm("k_proj", h).reshape(S, Hkv, Dh); v = mm("v_proj", h).reshape(S, Hkv, Dh)
    q = rope(rmsnorm(q, w["q_norm"]), c, s); k = rope(rmsnorm(k, w["k_norm"]), c, s)
    x = x + mm("o_proj", attn(q, k, v).reshape(S, Hq * Dh))
    h2 = rmsnorm(x, w["post_ln"]); g = mm("gate_proj", h2); u = mm("up_proj", h2)
    return (x + mm("down_proj", (g * (1.0 / (1.0 + np.exp(-g)))) * u)).astype(np.float32)

def agree(lb, lq):
    t = lb.argmax(-1); a1 = (t == lq.argmax(-1)).mean(); top5 = np.argsort(lq, -1)[:, -5:]
    return a1, np.mean([t[i] in top5[i] for i in range(len(t))])


if __name__ == "__main__":
    L, S = 28, 32
    print(f"DEFINITIVE FP4 rollout quality — REAL Qwen3-0.6B  L={L} S={S} head_dim={Dh} vocab=151936"); print("=" * 72)
    t0 = time.time(); W = load_qwen3(n_layers=L); print(f"  loaded in {time.time()-t0:.0f}s")
    rng = np.random.default_rng(0); ids = rng.integers(0, 151936, S)
    x0 = W["embed"][ids].astype(np.float32); c, s = cos_sin(S)
    configs = {"all BF16 (ref)": ({n: mm_bf16 for n in PROJS}, 0.0), "all MXFP8 (base)": (prec_cfg(set()), 0.0),
               "down FP4 +MXFP8": (prec_cfg({"down_proj"}), fp4_frac({"down_proj"})),
               "FFN  FP4 +MXFP8": (prec_cfg({"gate_proj", "up_proj", "down_proj"}), fp4_frac({"gate_proj", "up_proj", "down_proj"})),
               "all  FP4": (prec_cfg(set(PROJS), base=mm_nvfp4), 1.0)}
    fin = {}
    for nm, (P, _) in configs.items():
        x = x0.copy()
        for i in range(L): x = layer(x, W["layers"][i], c, s, P)
        fin[nm] = mm_bf16(rmsnorm(x, W["final_norm"]), W["lm_head"])
        print(f"  [{nm}] forward done")
    lb = fin["all BF16 (ref)"]
    print(f"  {'config':17s}{'FP4%FLOP':>9s}{'top-1':>9s}{'top-5':>9s}")
    for nm, (_, frac) in configs.items():
        a1, a5 = agree(lb, fin[nm]); print(f"  {nm:17s}{frac*100:8.0f}%{a1*100:8.1f}%{a5*100:8.1f}%")
    print("=" * 72)
