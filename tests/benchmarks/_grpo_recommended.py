"""Full GRPO step at the RECOMMENDED config: rollout B=128 (packed Md tile), training microbatched at
M=1024 (the real-vocab VRAM cap). Measures each phase + per-phase MFU. Real Qwen3-0.6B shapes, BF16.

  rollout : ResidentDecodeModel(Bp=128, maxS=S) — full S-step autoregressive decode (per-token grows with
            cache depth, so we time the FULL rollout, not a shallow extrapolation).
  training: ResidentModel(B=4, S) → M=1024 microbatch; the B=128 rollout's 128·S tokens train as
            (128·S/1024) grad-accum microbatches → train_time = n_micro·(fwd+bwd) + 1·step.

Run:  python tests/benchmarks/_grpo_recommended.py [G] [NUM_PROMPTS] [P] [GEN]
"""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

from ancora.model.qwen3_layer import TransformerLayer, Qwen3Config
from ancora.model.resident_model import ResidentModel
from ancora.model.resident_decode import ResidentDecodeModel

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
cfg = Qwen3Config()
H, Hq, Hkv, Dh, I = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate
qd, kd = Hq * Dh, Hkv * Dh
PEAK = 80e12
plp = lambda: H*qd + 2*H*kd + qd*H + 2*H*I + I*H          # per-layer matmul params
def log(m): print(m, flush=True)


def build(NL, V):
    rng = np.random.default_rng(0)
    return {"layers": [TransformerLayer(cfg, seed=i).w for i in range(NL)],
            "embed": (rng.standard_normal((V, H)) * 0.02).astype(np.float32),
            "final_norm": (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)}


def rollout(weights, B, V, NL, S):
    log(f"  [rollout] ResidentDecodeModel(Bp={B}, maxS={S}) — full {S}-step decode ...")
    dm = ResidentDecodeModel(cfg, weights, Bp=B, maxS=S, vocab=V)
    rng = np.random.default_rng(1)
    toks = [rng.integers(0, V, B).astype(np.int64) for _ in range(S)]
    for t in range(3):                                   # JIT warm (shallow)
        h = dm.decode_step(toks[t], t, si); dm._generate_step(h, si)
    cudart.cudaStreamSynchronize(si)
    t0 = time.perf_counter()
    for t in range(S):                                   # the real rollout (cache grows 0..S)
        h = dm.decode_step(toks[t], t, si); dm._generate_step(h, si)
    cudart.cudaStreamSynchronize(si)
    dt = time.perf_counter() - t0
    used = (cudart.cudaMemGetInfo()[2] - cudart.cudaMemGetInfo()[1]) / 1e9
    gemm = 2 * (NL * plp() + V * H) * B * S               # forward, B real rows × S steps
    attn = sum(2 * 2 * B * Hq * (t + 1) * Dh * NL for t in range(S))
    dm.free(); gc.collect(); cudart.cudaDeviceSynchronize()
    return dt, gemm + attn, used


def train_micro(weights, V, NL, S, Bmicro):
    M = Bmicro * S
    log(f"  [train] ResidentModel(B={Bmicro}, S={S}) → M={M} microbatch ...")
    m = ResidentModel(cfg, weights, Bmicro, S, vocab=V)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, (Bmicro, S)).astype(np.int64)
    lab = rng.integers(0, V, (Bmicro, S)).astype(np.int64)
    adv = rng.standard_normal(M).astype(np.float32)
    def fb():                                            # forward + backward (grad-accum body)
        h = m.forward(ids, si); m.loss_backward(h, lab, si, advantage=adv)
    for _ in range(2): fb(); m.step(si)
    cudart.cudaStreamSynchronize(si)
    t0 = time.perf_counter()
    for _ in range(3): fb()
    cudart.cudaStreamSynchronize(si); t_fb = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3): m.step(si)
    cudart.cudaStreamSynchronize(si); t_step = (time.perf_counter() - t0) / 3
    used = (cudart.cudaMemGetInfo()[2] - cudart.cudaMemGetInfo()[1]) / 1e9
    m.free(); gc.collect(); cudart.cudaDeviceSynchronize()
    return t_fb, t_step, used


def main(G=8, NUM_PROMPTS=16, P=128, GEN=128):
    NL, V = 28, 151936
    B = NUM_PROMPTS * G; S = P + GEN
    tok_total = B * S
    Mcap = 1024; Bmicro = max(1, Mcap // S); Mmicro = Bmicro * S
    n_micro = (tok_total + Mmicro - 1) // Mmicro
    log(f"RECOMMENDED GRPO step — Qwen3-0.6B  NL={NL} V={V}")
    log(f"  rollout B={B} (={NUM_PROMPTS}×G{G}), context S={S} (P={P}+GEN={GEN}) → {tok_total} tokens")
    log(f"  training microbatch M={Mmicro} (B={Bmicro}×S), grad-accum × {n_micro} microbatches")
    log("=" * 96)
    w = build(NL, V)

    tr, fr, ur = rollout(w, B, V, NL, S)
    t_fb, t_step, ut = train_micro(w, V, NL, S, Bmicro)
    tt = n_micro * t_fb + t_step                          # accumulate n_micro fwd+bwd, then ONE optimizer step

    f_layer = 6 * (NL * plp() + V * H) * tok_total
    f_attn = 3 * NL * (2 * 2 * B * Hq * S * S * Dh / 2)   # causal fwd ×3 (fwd+bwd), full batch
    ft = f_layer + f_attn

    mfu_r = fr / (tr * PEAK) * 100
    mfu_t = ft / (tt * PEAK) * 100
    total = tr + tt
    log("-" * 96)
    log(f"  ROLLOUT  : {tr*1e3:8.0f} ms   {fr/1e12:6.1f} TFLOP  MFU {mfu_r:4.1f}%   ({tr/S*1e3:.1f} ms/token avg, VRAM {ur:.1f} GB)")
    log(f"  TRAINING : {tt*1e3:8.0f} ms   {ft/1e12:6.1f} TFLOP  MFU {mfu_t:4.1f}%   "
        f"({n_micro}×{t_fb*1e3:.0f}ms fwd+bwd + {t_step*1e3:.0f}ms step, VRAM {ut:.1f} GB)")
    log("-" * 96)
    log(f"  FULL GRPO STEP: {total:.2f} s   rollout {tr/total*100:.0f}% : training {tt/total*100:.0f}%   "
        f"(training/rollout = {tt/tr:.1f}×)")
    log(f"  OVERALL MFU: {(fr+ft)/(total*PEAK)*100:.1f}%   ({tok_total} rollout-tokens + {tok_total} train-tokens / step)")
    log("=" * 96)


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 8, int(a[2]) if len(a) > 2 else 16,
         int(a[3]) if len(a) > 3 else 128, int(a[4]) if len(a) > 4 else 128)
