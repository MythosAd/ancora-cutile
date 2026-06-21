"""End-to-end GRPO step timing on REAL Qwen3-0.6B shapes (28 layers, V=151936), device-resident path.

Measures the two phases of one on-policy GRPO step and their split:
  • ROLLOUT  : ResidentDecodeModel — autoregressive decode (per-layer KV-cache, the ratio=1 engine).
               B = NUM_PROMPTS*G sequences decode in parallel in the Md=128 tile (B≤128 is "free").
  • TRAINING : ResidentModel — forward + loss_backward (policy-grad) + AdamW step over the B sequences.

VRAM-SAFE: phases are measured SEPARATELY (each model freed + device-synced before the next), and the
training model is skipped with a warning if it would exceed VRAM_GUARD of the card (avoids WDDM paging
thrash that looks like a hang on Windows). Rollout per-token cost is constant in B≤128, so it is measured
over a few steps and extrapolated to (P+GEN) — no need to run the full rollout. Progress is printed live.

Run:  python tests/benchmarks/bench_grpo_step.py [NUM_PROMPTS] [G] [P] [GEN] [--real]
"""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.resident_decode import ResidentDecodeModel
from ancora.kernels.attention import BKV

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config()
H, Hq, Hkv, Dh, I = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
qd, kd = Hq * Dh, Hkv * Dh
PEAK_BF16 = 80e12       # RTX 5080 Laptop real BF16+f32-acc ceiling ([[gemm-mfu-ceiling]])
VRAM_GUARD = 0.90       # don't build a model whose footprint would exceed this fraction of the card


def vram():
    err, free, total = cudart.cudaMemGetInfo()
    return (total - free) / 1e9, free / 1e9, total / 1e9


def log(msg):
    print(msg, flush=True)


def per_layer_params():
    return H * qd + 2 * H * kd + qd * H + 2 * H * I + I * H


def attn_fwd_flops(B, S):
    """Causal self-attention forward FLOPs, ONE layer (QK^T + P·V, MAC=2, causal halves)."""
    return 2 * 2 * B * Hq * S * S * Dh / 2


def build_weights(NL, V, real=False):
    if real:
        from ancora.model.load_qwen3 import load_qwen3
        w = load_qwen3(n_layers=NL, want_lm_head=False)
        log(f"  loaded REAL Qwen3-0.6B: {len(w['layers'])} layers, embed {w['embed'].shape}")
        return w
    rng = np.random.default_rng(123)
    return {"layers": [TransformerLayer(cfg, seed=i).w for i in range(NL)],
            "embed": (rng.standard_normal((V, H)) * 0.02).astype(np.float32),
            "final_norm": (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)}


def time_rollout(weights, B, V, NL, P, GEN):
    """Per-decode-token time (decode_step + boundary) over a few steps → extrapolate to (P+GEN).
    Per-token cost is constant for B≤128 (Md=128 tile), so a short measurement is faithful and fast."""
    probe_steps = 2 * BKV                      # 128 steps is plenty; maxS multiple of BKV
    log(f"  [rollout] building ResidentDecodeModel (Bp={B}, maxS={probe_steps}) ...")
    dm = ResidentDecodeModel(cfg, weights, Bp=B, maxS=probe_steps, vocab=V)
    used, free, tot = vram()
    log(f"  [rollout] built, VRAM {used:.1f}/{tot:.1f} GB; warming up (JIT) ...")
    rng = np.random.default_rng(1)
    toks = [rng.integers(0, V, B).astype(np.int64) for _ in range(probe_steps)]
    for t in range(4):                         # JIT warmup
        h = dm.decode_step(toks[t], t, si); dm._generate_step(h, si)
    cudart.cudaStreamSynchronize(si)
    NSTEP = 24
    log(f"  [rollout] timing {NSTEP} decode tokens (greedy, device argmax) ...")
    t0 = time.perf_counter()
    for t in range(NSTEP):
        h = dm.decode_step(toks[t], t, si); dm._generate_step(h, si)
    cudart.cudaStreamSynchronize(si)
    per_tok = (time.perf_counter() - t0) / NSTEP
    dm.free(); gc.collect(); cudart.cudaDeviceSynchronize()
    steps = P + GEN
    dt = per_tok * steps
    gemm = 2 * (NL * per_layer_params() + V * H) * B * steps
    attn = sum(2 * 2 * B * Hq * (t + 1) * Dh * NL for t in range(steps))
    return dt, per_tok, gemm + attn, used


def time_train(weights, B, S, V, NL):
    M = B * S
    # footprint estimate: weights+AdamW (14 B/param) + boundary (M·V·8) + per-layer fwd+bwd activations
    # (~260 KB/token/layer, calibrated: M=1024 → ~7.5 GB activations over 28 layers).
    est = (596e6 * 14 + M * V * 8 + M * NL * 260e3) / 1e9
    used0, free0, tot = vram()
    if est > VRAM_GUARD * tot:
        log(f"  [train] SKIP — estimated footprint {est:.1f} GB > {VRAM_GUARD*100:.0f}% of {tot:.1f} GB "
            f"(would thrash WDDM). Lower S/B or use chunked-CE.")
        return None, 0.0, 0.0
    log(f"  [train] building ResidentModel (M={M}, est {est:.1f} GB) ...")
    m = ResidentModel(cfg, weights, B, S, vocab=V)
    used, free, tot = vram()
    log(f"  [train] built, VRAM {used:.1f}/{tot:.1f} GB; warming up ...")
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, (B, S)).astype(np.int64)
    labels = rng.integers(0, V, (B, S)).astype(np.int64)
    adv = rng.standard_normal(M).astype(np.float32)
    def step():
        h = m.forward(ids, si); m.loss_backward(h, labels, si, advantage=adv); m.step(si)
    for _ in range(2): step()
    cudart.cudaStreamSynchronize(si)
    log(f"  [train] timing 3 steps ...")
    t0 = time.perf_counter()
    for _ in range(3): step()
    cudart.cudaStreamSynchronize(si)
    dt = (time.perf_counter() - t0) / 3
    gemm = 6 * (NL * per_layer_params() + V * H) * M
    attn = 3 * NL * attn_fwd_flops(B, S)
    m.free(); gc.collect(); cudart.cudaDeviceSynchronize()
    return dt, gemm + attn, used


def main(NUM_PROMPTS=2, G=8, P=64, GEN=64, real=False):
    NL = 28; V = 151936
    B = NUM_PROMPTS * G; S = P + GEN; M = B * S
    used, free, tot = vram()
    log(f"GRPO step — {'REAL' if real else 'random@real-shape'} Qwen3-0.6B  NL={NL} H={H} V={V}")
    log(f"  {NUM_PROMPTS} prompts × G={G} = {B} seqs, P={P}+GEN={GEN}=S={S}, train M={M}  |  "
        f"VRAM {tot:.1f} GB, 596M params")
    log("=" * 92)
    weights = build_weights(NL, V, real=real)

    tr, per_tok, fr, ur = time_rollout(weights, B, V, NL, P, GEN)
    tt, ft, ut = time_train(weights, B, S, V, NL)

    log("-" * 92)
    mfu_r = fr / (tr * PEAK_BF16) * 100
    log(f"  ROLLOUT  ({P+GEN} tokens × B={B}):  {tr*1e3:8.1f} ms  ({per_tok*1e3:.1f} ms/token, constant in B≤128)  "
        f"{fr/1e9:7.0f} GFLOP  MFU {mfu_r:4.1f}%   {B/per_tok:.0f} tok/s aggregate   VRAM {ur:.1f} GB")
    if tt is None:
        log(f"  TRAINING (M={M}): skipped (VRAM). Use a smaller microbatch; rollout is the dominant phase anyway.")
        log("=" * 92); return
    mfu_t = ft / (tt * PEAK_BF16) * 100
    total = tr + tt
    log(f"  TRAINING (fwd+bwd+AdamW, M={M}): {tt*1e3:8.1f} ms  {ft/1e9:7.0f} GFLOP  MFU {mfu_t:4.1f}%   VRAM {ut:.1f} GB")
    log("-" * 92)
    log(f"  FULL GRPO STEP: {total*1e3:.0f} ms   rollout {tr/total*100:.0f}% : training {tt/total*100:.0f}%   "
        f"(rollout/training = {tr/tt:.1f}×)")
    log(f"  OVERALL MFU: {(fr+ft)/(total*PEAK_BF16)*100:.1f}%")
    log("=" * 92)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    real = "--real" in sys.argv
    main(int(a[0]) if len(a) > 0 else 2, int(a[1]) if len(a) > 1 else 8,
         int(a[2]) if len(a) > 2 else 64, int(a[3]) if len(a) > 3 else 64, real=real)
