"""Tied-embedding device LM-head — validate the device tied-head path in ISOLATION before wiring
into ResidentModel. With tie_word_embeddings, lm_head.weight == embed (V,H), so:
    logits        = hidden @ embed.T            via _gemm_nt_f32(hidden, embed)
    CE            (_ce_stats / _ce_grad)         → logprob, lse, glogit(M,V) bf16
    dhidden       = glogit @ embed               via _gemm(glogit, embed)
    embed_head_g  = glogitᵀ @ hidden             via _gemm_dW(glogit, hidden)   (V,H)
    input gather  x0 = onehot @ embed            via _gemm(onehot, embed)        (== embed[ids])
    input_embed_g = onehotᵀ @ d_x0               via _gemm_dW(onehot, d_x0)       (V,H)
    embed_grad    = embed_head_g + input_embed_g via _acc_f32
Compared vs numpy fp64 AND the proven host linear_ce(hidden, embed.T)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa
from ancora.kernels.loss import (_gemm, _ce_stats, _ce_grad, linear_ce, _GpuArray as G,
                                  f32_to_bf16_bits as f32bf, GTM, GTN, GTK, CTM, TV)
from ancora.kernels.fused import _gemm_nt_f32, _gemm_dW, _acc_f32, ACM, ACN

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
_bf = lambda x: bf32(f32bf(x))
rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

M, H, V = 128, 256, 2048           # small but tile-divisible (128|M,V; 64|H; 128|H)
rng = np.random.default_rng(0)
hidden = _bf((rng.standard_normal((M, H)) * 0.3).astype(np.float32))
embed  = _bf((rng.standard_normal((V, H)) * 0.05).astype(np.float32))   # (V,H) tied param
labels = rng.integers(0, V, M).astype(np.int32)
ids    = rng.integers(0, V, M).astype(np.int32)
adv    = np.ones(M, np.float32)     # SFT
invM   = 1.0 / M

# ── numpy fp64 reference ────────────────────────────────────────────────────
hd, ed = hidden.astype(np.float64), embed.astype(np.float64)
logits_r = hd @ ed.T                                            # (M,V)
mx = logits_r.max(-1, keepdims=True)
lse_r = (mx + np.log(np.exp(logits_r - mx).sum(-1, keepdims=True)))[:, 0]
logprob_r = logits_r[np.arange(M), labels] - lse_r
sm = np.exp(logits_r - lse_r[:, None])
oneh = np.zeros((M, V)); oneh[np.arange(M), labels] = 1.0
glogit_r = invM * adv[:, None] * (sm - oneh)                     # (M,V)
dhidden_r = glogit_r @ ed                                        # (M,H)
embed_head_g_r = glogit_r.T @ hd                                 # (V,H)
x0_r = ed[ids]                                                  # (M,H) gather
d_x0 = _bf((rng.standard_normal((M, H)) * 0.1).astype(np.float32))   # arbitrary grad wrt x0
onehot_in = np.zeros((M, V), np.float32); onehot_in[np.arange(M), ids] = 1.0
input_embed_g_r = onehot_in.T.astype(np.float64) @ _bf(d_x0).astype(np.float64)   # (V,H)
embed_grad_r = embed_head_g_r + input_embed_g_r

# ── device tied-head path ────────────────────────────────────────────────────
gh = G(f32bf(hidden)); ge = G(f32bf(embed)); glab = G(labels.reshape(M, 1)); gadv = G(adv.reshape(M, 1))
gLg = G(np.zeros((M, V), np.float32)); gp = G(np.zeros((M, 1), np.float32)); gls = G(np.zeros((M, 1), np.float32))
gG = G(np.zeros((M, V), np.uint16))

# logits = hidden @ embed.T
ct.launch(si, (M // 128, V // 128, 1), _gemm_nt_f32, (gh, ge, gLg, H // 64, 128, 128, 64))
ct.launch(si, (M // CTM, 1, 1), _ce_stats, (gLg, glab, gp, gls, V // TV))
ct.launch(si, (M // CTM, 1, 1), _ce_grad, (gLg, gls, glab, gadv, gG, V // TV, invM))
cudart.cudaStreamSynchronize(si)
logits_d = gLg.to_numpy(); logprob_d = gp.to_numpy().reshape(M); glogit_d = bf32(gG.to_numpy())

# dhidden = glogit @ embed  (glogit (M,V) bf16, embed (V,H) bf16 → (M,H))
gdh = G(np.zeros((M, H), np.float32))
ct.launch(si, (M // GTM, H // GTN, 1), _gemm, (gG, ge, gdh, V // GTK, GTM, GTN, GTK))
# embed_head_grad = glogitᵀ @ hidden  (V,H) via _gemm_dW(x=glogit, dy=hidden)
T = 64
gehg = G(np.zeros((V, H), np.float32))
ct.launch(si, (V // T, H // T, 1), _gemm_dW, (gG, gh, gehg, M // T, T, T, T))
# input gather x0 = onehot @ embed  (M,H) f32
goh = G(f32bf(onehot_in)); gx0 = G(np.zeros((M, H), np.float32))
ct.launch(si, (M // GTM, H // GTN, 1), _gemm, (goh, ge, gx0, V // GTK, GTM, GTN, GTK))
# input_embed_grad = onehotᵀ @ d_x0  (V,H)
gdx0 = G(f32bf(d_x0)); gieg = G(np.zeros((V, H), np.float32))
ct.launch(si, (V // T, H // T, 1), _gemm_dW, (goh, gdx0, gieg, M // T, T, T, T))
# embed_grad = embed_head_grad + input_embed_grad  (in place into gehg)
ct.launch(si, (V // ACM, H // ACN, 1), _acc_f32, (gieg, gehg))
cudart.cudaStreamSynchronize(si)
dhidden_d = gdh.to_numpy(); ehg_d = gehg.to_numpy(); x0_d = gx0.to_numpy()

# ── host linear_ce cross-check (uses w_head=(H,V) = embed.T) ──────────────────
lp_h, dh_h, dW_h = linear_ce(hidden, np.ascontiguousarray(embed.T), labels, si, advantage=adv)

print("Tied-embedding device LM-head — vs numpy fp64 (+ host linear_ce)")
print("=" * 64)
r = {}
r["logits"]       = rel(logits_d, logits_r)
r["logprob"]      = rel(logprob_d, logprob_r)
r["glogit"]       = rel(glogit_d, glogit_r)
r["dhidden"]      = rel(dhidden_d, dhidden_r)
r["embed_head_g"] = rel(ehg_d, embed_grad_r)       # ehg_d now holds the SUM (head + input)
r["x0 gather"]    = rel(x0_d, x0_r)
r["host logprob"] = rel(logprob_d, lp_h)            # device CE == host linear_ce CE
r["host dhidden"] = rel(dhidden_d, dh_h)            # device dhidden == host (glogit@embed == glogit@w_headᵀ)
ok = True
for k, v in r.items():
    thr = 0.02 if "glogit" not in k else 0.05
    o = v < thr; ok &= o
    print(f"  {k:14s}: {v*100:6.3f}%  {'OK' if o else 'FAIL'}")
# x0 gather must be near-exact (onehot@embed == embed[ids], one term)
print("-" * 64)
print(f"  {'PASS — tied device head matches numpy + host linear_ce' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
