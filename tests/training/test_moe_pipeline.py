"""Full MoE pipeline: train (HybridOptimizer Muon+AdamW) → infer (prefill logits → argmax),
with per-stage wall-time + achieved-TFLOPS (MFU proxy). Identifies training/inference bottlenecks."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig, layer_schedule
from ancora.model.moe_model import MoEModel
from ancora.optim.hybrid import HybridOptimizer

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])


def active_params(cfg):
    """Active params/token = embed-lookup(0 flops here) + attn + dense-FFN + top-k experts."""
    H, Hq, Hkv, Dh = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    attn = H*Hq*Dh + 2*H*Hkv*Dh + Hq*Dh*H
    swig = lambda i: 3*H*i
    nd = cfg.n_layers // 2
    return cfg.n_layers*attn + nd*swig(cfg.dense_inter) + nd*(cfg.top_k*swig(cfg.expert_inter) + H*cfg.n_experts)


def main():
    cfg = MoEConfig(vocab=8192, n_layers=4)
    print(f"MoEModel  H={cfg.hidden} L={cfg.n_layers} E={cfg.n_experts}/top{cfg.top_k} "
          f"V={cfg.vocab}  schedule={layer_schedule(cfg)}")
    model = MoEModel(cfg, seed=0, grouped=True, tie=True)
    opt = HybridOptimizer(model.params())
    print("optimizer routing: Muon=%d params, AdamW=%d params" %
          (len(opt.routing()["muon"]), len(opt.routing()["adamw"])))

    B, S = 4, 128; M = B * S
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab, (B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (M,)).astype(np.int64)
    Aflops = 6 * active_params(cfg) * M     # fwd+bwd ≈ 6·active·tokens

    # ── TRAINING ──
    print("\n--- training (HybridOptimizer) ---")
    ce0 = None; tstep = []
    for step in range(12):
        t0 = time.perf_counter()
        h, cache = model.forward(ids, si)
        ce, grads = model.loss_backward(h, labels, cache, si)
        opt.step(grads, si)
        model.set_weights(opt.weights())
        dt = time.perf_counter() - t0; tstep.append(dt)
        if ce0 is None: ce0 = ce
        if step % 3 == 0 or step == 11:
            print(f"  step {step:2d}: CE={ce:.4f}  {dt*1e3:.0f} ms  ({Aflops/dt/1e12:.2f} TFLOP/s active)")
    med = sorted(tstep)[len(tstep)//2]
    print(f"  median step {med*1e3:.0f} ms → {Aflops/med/1e12:.2f} TFLOP/s (fwd+bwd, active params)")

    # ── INFERENCE (prefill → logits → argmax) ──
    print("\n--- inference (prefill forward → logits) ---")
    t0 = time.perf_counter()
    h, _ = model.forward(ids, si)
    logits = h @ model._head()                       # (M, V)
    dt = time.perf_counter() - t0
    pred = logits.argmax(-1)
    acc = float((pred == labels).mean())
    Iflops = 2 * active_params(cfg) * M              # fwd ≈ 2·active·tokens
    print(f"  prefill {dt*1e3:.0f} ms  ({Iflops/dt/1e12:.2f} TFLOP/s)  argmax-acc(trained batch)={acc*100:.0f}%")
    print(f"  CE {ce0:.3f} → {ce:.3f}")
    return ce < ce0 * 0.5


if __name__ == "__main__":
    print("=" * 72)
    ok = main()
    print("=" * 72)
    print("  pipeline OK" if ok else "  pipeline ran (CE not halved — see notes)")
