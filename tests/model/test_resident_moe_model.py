"""ResidentMoEModel end-to-end SFT (milestone 3 capstone): the full device-resident interleaved
dense/MoE + local/global model — embed gather, scheduled ResidentMoEDense/MoE layers, tied embed/
LM-head boundary, device AdamW everywhere. Overfits a fixed (input_ids, labels) batch and watches
CE collapse; checks forward determinism (bitwise) too. Small vocab so the boundary fits VRAM.
Foreground only."""
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

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
stream_obj = dev.create_stream()
si = int(stream_obj.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)


def main():
    # small vocab + a few layers covering all 4 cell types (local/global × dense/MoE)
    cfg = MoEConfig(vocab=2048, n_layers=4, period=2)     # sched: (L,D)(G,M)(L,D)(G,M)
    B, S = 2, 128; M = B * S
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    print("schedule (is_global, ffn_dense):", host.sched)
    model = ResidentMoEModel(cfg, from_host(host, B, S), B, S)

    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, size=(B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, size=(M,)).astype(np.int64)

    # determinism: same forward twice → bitwise-identical hidden
    h1 = model.forward(ids, si); sync(); h1 = h1.copy()
    h2 = model.forward(ids, si); sync()
    det = np.abs(h1.astype(np.float64) - h2.astype(np.float64)).max()
    print(f"forward determinism max|Δ| = {det:.3e}  ({'bitwise' if det == 0 else 'NONDET'})")

    # overfit. NB random-init + tied-embed (input & output share one moving weight) + per-step MoE
    # routing drift ⇒ slow convergence (real Qwen3 weights collapse in ~4 steps, [[real-qwen3-model]]).
    # The capstone proves the full device-resident stack TRAINS end-to-end: bitwise-deterministic
    # forward + a substantial monotone CE drop through embed→4×(dense/MoE, local/global)→tied head.
    ce0 = None; cemin = 1e9; steps = 150
    t0 = time.perf_counter()
    for it in range(steps):
        h = model.forward(ids, si)
        ce = model.loss_backward(h, labels, si)
        model.step(si, lr=2e-3)
        sync()
        if it == 0: ce0 = ce
        cemin = min(cemin, ce)
        if it in (0, 50, 100, steps - 1):
            print(f"  step {it:3d}: CE {ce:.4f}")
    dt = (time.perf_counter() - t0) / steps
    ok = det == 0 and cemin < 0.4 * ce0
    print(f"  CE {ce0:.3f} -> min {cemin:.3f} ({cemin/ce0*100:.1f}%)   {dt*1e3:.0f} ms/step (NL={cfg.n_layers})")
    print("=" * 70)
    print("  PASS (resident MoE model trains: CE collapses + forward bitwise-deterministic)"
          if ok else f"  FAIL (det={det}, CE {ce0:.3f}->min {cemin:.3f})")
    return ok


if __name__ == "__main__":
    print("ResidentMoEModel end-to-end SFT (full device-resident dense/MoE + local/global)")
    print("=" * 70)
    main()
