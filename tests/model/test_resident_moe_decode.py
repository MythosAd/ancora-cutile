"""ResidentMoEDecodeModel — the MoE-family ROLLOUT engine, now with ALL performance levers:
device-position kernels (pos in device memory) → ONE-token CUDA-graph replay; local-layer RING
KV-cache (O(window) memory — sized to wrap during the test); Gumbel-max device sampling; device
closed loop (one sync per rollout); the DECODE MEGAKERNEL fusions (2026-06-12: DTN=32 GEMMs,
o/down GEMM+residual epilogues, fused MoE gate+up+SwiGLU, one-pass pick+CE, Bp-row norms/rope —
every fusion probed BITWISE, see _probe_decode_{tiles,fused,attn}.py; 8.6→5.4 ms/step graph).

  (A) TEACHER-FORCED ratio=1 vs SFT training (every position; ring WRAPS at pos ≥ 256).
  (B) GREEDY rollout lp BITWISE == prefix-GRPO trainer lp (boundary dup-row + suffix).
  (C) ZERO-COPY weight sharing through a GRPO step.
  (F) GRAPH replay == direct launches BITWISE (ids + lp), greedy AND sampled.
  (G) SAMPLING: deterministic given seed, diverse across seeds, sampled-token lp BITWISE ==
      prefix-trainer score of those tokens (the GRPO closure holds for sampled rollouts).
  (D) throughput: direct vs graph; real-size bench in bench_real(). Foreground only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host
from ancora.model.resident_prefix_model import ResidentPrefixMoEModel
from ancora.model.resident_moe_decode import ResidentMoEDecodeModel

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)


def main():
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2, window=128)   # (L,D)(G,M)(L,D)(G,M)
    Sp, Sc, G = 256, 64, 4; S = Sp + Sc                             # S=320 > ring rows 256 → wraps
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, S)
    rng = np.random.default_rng(3)

    # ── (A) teacher-forced vs the SFT trainer (ring wrap exercised at pos ≥ 256) ──
    train = ResidentMoEModel(cfg, w, G, S, device_route=True)
    eng = ResidentMoEDecodeModel(train, Bp=G, maxS=S, si=si)
    ids = rng.integers(0, cfg.vocab, size=(G, S)).astype(np.int64)
    labels = np.concatenate([ids[:, 1:], np.zeros((G, 1), np.int64)], 1)
    lps = eng.score(ids, labels, si)
    train.forward(ids, si)
    train.loss_backward(None, labels.reshape(-1), si)
    lpt = train.glp.to_numpy().reshape(G, S)
    eA = float(np.abs(lps - lpt).max())
    print(f"  (A) teacher-forced decode vs SFT-training lp (ring wraps): Δ={eA:.0e}  "
          f"{'OK (bitwise → ratio=1)' if eA == 0 else 'FAIL'}")

    # ── (B) greedy generate vs the PREFIX GRPO trainer ──
    prefix = ResidentPrefixMoEModel(cfg, w, Sp, Sc, G, device_route=True)
    eng2 = ResidentMoEDecodeModel(prefix, Bp=G, maxS=S, si=si)
    prompt = rng.integers(0, cfg.vocab, size=(Sp,)).astype(np.int64)
    prompts = np.tile(prompt, (G, 1))
    gen, glp = eng2.generate(prompts, Sc, si)
    prefix.forward_prefix(prompt, gen, si)
    prefix.grpo_loss_backward(None, gen, np.ones(G, np.float32), si)
    lpp = prefix.glp.to_numpy().reshape(prefix.Mh)
    eB = 0.0
    for i in range(G):
        eB = max(eB, float(abs(lpp[prefix.M + i] - glp[i, 0])))
        eB = max(eB, float(np.abs(lpp[Sp + i * Sc: Sp + (i + 1) * Sc - 1] - glp[i, 1:]).max()))
    print(f"  (B) greedy rollout lp vs prefix-GRPO-training lp: Δ={eB:.0e}  "
          f"{'OK (bitwise → no importance sampling)' if eB == 0 else 'FAIL'}")

    # ── (F) graph replay == direct, greedy AND sampled ──
    gen_g, glp_g = eng2.generate(prompts, Sc, si, so=so, dev=dev, use_graph=True)
    eF1 = int(np.abs(gen_g - gen).max()) + float(np.abs(glp_g - glp).max())
    smp_d, slp_d = eng2.generate(prompts, Sc, si, sample=True, temperature=1.0, seed=7)
    smp_g, slp_g = eng2.generate(prompts, Sc, si, so=so, dev=dev, use_graph=True,
                                 sample=True, temperature=1.0, seed=7)
    eF2 = int(np.abs(smp_g - smp_d).max()) + float(np.abs(slp_g - slp_d).max())
    print(f"  (F) graph vs direct: greedy Δ={eF1:.0e}  sampled Δ={eF2:.0e}  "
          f"{'OK (bitwise replay)' if eF1 == 0 and eF2 == 0 else 'FAIL'}")

    # ── (G) sampling: seed-deterministic, seed-diverse, lp == trainer score ──
    smp_d2, _ = eng2.generate(prompts, Sc, si, sample=True, temperature=1.0, seed=7)
    det = int(np.abs(smp_d2 - smp_d).max())
    smp_o, _ = eng2.generate(prompts, Sc, si, sample=True, temperature=1.0, seed=8)
    div = float((smp_o != smp_d).mean())
    prefix.forward_prefix(prompt, smp_d, si)
    prefix.grpo_loss_backward(None, smp_d, np.ones(G, np.float32), si)
    lpp = prefix.glp.to_numpy().reshape(prefix.Mh)
    eG = 0.0
    for i in range(G):
        eG = max(eG, float(abs(lpp[prefix.M + i] - slp_d[i, 0])))
        eG = max(eG, float(np.abs(lpp[Sp + i * Sc: Sp + (i + 1) * Sc - 1] - slp_d[i, 1:]).max()))
    okG = det == 0 and div > 0.5 and eG == 0.0
    print(f"  (G) sampled: seed-det Δ={det}  cross-seed diversity {div*100:.0f}%  "
          f"lp vs trainer Δ={eG:.0e}  {'OK' if okG else 'FAIL'}")

    # ── (C) zero-copy weight sharing through a GRPO step ──
    r = np.zeros(G); r[0] = 1.0
    adv = ((r - r.mean()) / (r.std() + 1e-6)).astype(np.float32)
    prefix.step(si, lr=2e-3); sync()
    prefix.forward_prefix(prompt, gen, si)
    prefix.grpo_loss_backward(None, gen, adv, si)
    lpp2 = prefix.glp.to_numpy().reshape(prefix.Mh)
    full = np.concatenate([prompts, gen], 1)
    labf = np.concatenate([full[:, 1:], np.zeros((G, 1), np.int64)], 1)
    lps2 = eng2.score(full, labf, si)
    eC = 0.0
    for i in range(G):
        eC = max(eC, float(abs(lpp2[prefix.M + i] - lps2[i, Sp - 1])))
        eC = max(eC, float(np.abs(lpp2[Sp + i * Sc: Sp + (i + 1) * Sc - 1] - lps2[i, Sp:S - 1]).max()))
    print(f"  (C) post-step (shared weights, zero copy) decode vs trainer lp: Δ={eC:.0e}  "
          f"{'OK (bitwise)' if eC == 0 else 'FAIL'}")

    # ── (D) throughput: direct vs graph ──
    ntok = Sp + Sc - 1
    t0 = time.perf_counter(); reps = 3
    for _ in range(reps): eng2.generate(prompts, Sc, si)
    dt = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps): eng2.generate(prompts, Sc, si, so=so, dev=dev, use_graph=True)
    dg = (time.perf_counter() - t0) / reps
    print(f"  (D) Bp={G}: direct {dt/ntok*1e3:.2f} ms/step vs GRAPH {dg/ntok*1e3:.2f} ms/step "
          f"= {dt/dg:.1f}x  ({G*ntok/dg:.0f} tok/s, NL={cfg.n_layers}, V={cfg.vocab})")

    return eA == 0 and eB == 0 and eF1 == 0 and eF2 == 0 and okG and eC == 0


def bench_real():
    """Real-size decode bench: full vocab, 12 layers, Bp=32 — tok/s, HW-MFU and weight-bandwidth
    utilization, direct vs graph. (Decode is weight-BW bound; batch amortizes the weight reads.)"""
    cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)  # the real schedule (5L:1G, D/M)
    Bp, P, NEW = 32, 512, 64; maxS = 1024
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, 128)
    train = ResidentMoEModel(cfg, w, 1, 128, device_route=True)       # minimal trainer (weight host)
    eng = ResidentMoEDecodeModel(train, Bp=Bp, maxS=maxS, si=si)
    rng = np.random.default_rng(0)
    prompts = rng.integers(0, cfg.vocab, size=(Bp, P)).astype(np.int64)
    eng.generate(prompts, 4, si)                                      # warm (JIT all kernels)
    eng.generate(prompts, 4, si, so=so, dev=dev, use_graph=True)
    ntok = P + NEW - 1
    t0 = time.perf_counter(); eng.generate(prompts, NEW, si); dt = (time.perf_counter() - t0) / ntok
    t0 = time.perf_counter(); eng.generate(prompts, NEW, si, so=so, dev=dev, use_graph=True)
    dg = (time.perf_counter() - t0) / ntok
    # FLOPs/step (Md=128 rows through every GEMM) + weight bytes/step (decode reads them all)
    H, V, Ie, E, Md = cfg.hidden, cfg.vocab, cfg.expert_inter, cfg.n_experts, 128
    qd, kd = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    p_attn = H * qd + 2 * H * kd + qd * H
    p_dense, p_moe = 3 * H * H, cfg.top_k * 3 * H * Ie                # active expert params
    nl_d = sum(1 for l in eng.layers if l.ffn_dense); nl_m = cfg.n_layers - nl_d
    flops = 2 * Md * (cfg.n_layers * p_attn + nl_d * p_dense + nl_m * p_moe + V * H)
    wbytes = 2 * (cfg.n_layers * p_attn + nl_d * p_dense + nl_m * (3 * E * H * Ie) + V * H)
    print(f"  REAL SIZE (NL=12, V={V}, Bp={Bp}, P={P}): direct {dt*1e3:.2f} → graph {dg*1e3:.2f} ms/step "
          f"({dt/dg:.1f}x) = {Bp/dg:.0f} tok/s")
    print(f"    graph step: {flops/dg/1e12:.1f} TFLOPS hw-MFU {flops/dg/210e12*100:.1f}% "
          f"(useful ×{Bp}/128) | weight-BW {wbytes/dg/1e9:.0f} GB/s = {wbytes/dg/896e9*100:.0f}% peak")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":     # child: real-size bench with the WHOLE GPU
        bench_real(); sys.exit(0)                        # (any other live model oversubscribes 16GB
    import subprocess                                    #  → WDDM paging, 11→120 ms/step)
    print("ResidentMoEDecodeModel — all perf levers: device-pos GRAPH, ring KV, sampling, batch")
    print("=" * 94)
    subprocess.run([sys.executable, __file__, "bench"])  # BEFORE main() — the parent holds no VRAM yet
    ok = main()
    print("=" * 94)
    print("  PASS (rollout==training bitwise incl. ring-wrap/graph/sampled; one sync per rollout)"
          if ok else "  FAIL")
