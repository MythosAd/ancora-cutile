"""Muon-vs-AdamW OPTIMIZER STEP time at real size (task #31) + NS batching granularity.

Answers two questions:
  (1) can the fully-batched Muon step beat the AdamW step? (honest FLOP math: the NS chains add
      ~12 TFLOP of GEMM per step at NL=12 — AdamW is a pure BW sweep with ZERO matmuls, so Muon
      can only approach it, never beat it, unless AdamW is paging);
  (2) the PIPELINE-PARALLEL granularity: global batch NS vs per-2-layer vs per-1-layer buckets
      (what a pipeline stage would own) vs per-weight — measures the launch/underfill cost of
      finer scoping.

ONE config per process (co-resident models oversubscribe 16 GB → silent WDDM paging, CLAUDE.md):
    python _bench_muon_vs_adamw.py adamw|global|scope2|scope1|perweight [M]
Run each mode in its own fresh invocation and compare the printed numbers."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)


def timed(fn, n=10):
    fn(); sync()                                             # warm (JIT + first-touch)
    best = 1e9
    for _ in range(3):                                       # best-of-3 windows (clock bimodality)
        t = time.perf_counter()
        for _ in range(n): fn()
        sync(); best = min(best, (time.perf_counter() - t) / n * 1e3)
    return best


def main(mode, M=2048):
    cfg = MoEConfig(vocab=151936, n_layers=12, period=6, window=512)
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, 1, M)
    kw = dict(adamw=dict(optimizer="adamw"),
              global_=dict(optimizer="muon"),
              scope2=dict(optimizer="muon", muon_scope=2),
              scope1=dict(optimizer="muon", muon_scope=1),
              perweight=dict(optimizer="muon", batch_proj=False))[mode if mode != "global" else "global_"]
    m = ResidentMoEModel(cfg, w, 1, M, device_route=True, **kw)
    ids = np.random.default_rng(0).integers(0, cfg.vocab, (1, M)).astype(np.int64)
    labels = np.random.default_rng(1).integers(0, cfg.vocab, (M,)).astype(np.int64)
    h = m.forward(ids, si); m.loss_backward(h, labels, si); sync()   # populate grads + warm

    free0 = cudart.cudaMemGetInfo()[1]
    t_step = timed(lambda: m.step(si, lr=1e-4))
    parts = {}
    if m.batched_proj_muon is not None:
        parts["proj NS (batched)"] = timed(lambda: m.batched_proj_muon.step(si, m.muon_lr))
        nch = sum((len(ws) + m.batched_proj_muon.scr[k]["c"] - 1) // m.batched_proj_muon.scr[k]["c"]
                  for k, ws in m.batched_proj_muon.groups.items())
        parts["proj NS (batched)"] = (parts["proj NS (batched)"],
                                      f"{sum(len(v) for v in m.batched_proj_muon.groups.values())} weights, "
                                      f"{len(m.batched_proj_muon.groups)} groups, {nch} chains")
    t_layers = timed(lambda: [l.step(si, 1e-4, 0.9, 0.999, 1e-8, 0.0, m.muon_lr) for l in m.layers])
    moe_layers = [l for l in m.layers if hasattr(l, "moe")]
    t_experts = timed(lambda: [l.moe.step(si, 1e-4, muon_lr=m.muon_lr) for l in moe_layers])
    parts["layer step (norms/experts…)"] = (t_layers, "")
    parts["  └ expert NS"] = (t_experts, f"{len(moe_layers)} MoE layers × 3 chains E=16")

    print(f"[{mode}] M={M} NL={cfg.n_layers} V={cfg.vocab}")
    print(f"  optimizer step total : {t_step:7.1f} ms   (GPU mem in use ≈ {(cudart.cudaMemGetInfo()[1] and (24 - free0/1e9)):.1f} GB)" if False else
          f"  optimizer step total : {t_step:7.1f} ms")
    for name, v in parts.items():
        t, note = v if isinstance(v, tuple) else (v, "")
        print(f"    {name:28s}: {t:7.1f} ms   {note}")
    print(f"    embed/norm AdamW ≈ step − proj − layers = "
          f"{t_step - sum(v[0] for k, v in parts.items() if not k.startswith('  ')):7.1f} ms")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "adamw"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    main(mode, M)
