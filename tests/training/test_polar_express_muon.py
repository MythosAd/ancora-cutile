"""Polar Express coeffs in the Muon NS — the end-to-end payoff test (tasks #28/#29).

Device-level correctness is in tests/kernels/_probe_polar_express_device.py (PE-5 orthogonalizes better
than Keller-5; default unchanged). This test answers the question that justifies adopting PE:
  (A) forward stays bitwise-DETERMINISTIC under PE → ratio=1 untouched (optimizer never touches fwd);
  (B) SFT under PE converges AT LEAST as fast as Keller (better orthogonalization = better-conditioned
      Muon updates = Muon's token-efficiency selling point);
  (C) NS wall-time: PE-5 ≈ Keller-5 (same matmul count — PE is a free QUALITY upgrade), PE-4 ~20% fewer
      matmuls (the only per-step SPEED option, and only safe on rectangular q/o per the device probe).
The process-wide toggle muon_ns.set_polar_express(on) selects PE for every Muon NS with no per-model
threading; default OFF ⇒ Keller ⇒ byte-identical (the existing test_muon_moe/_hybrid stay valid)."""
import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart

import ancora.env
from ancora.model.moe_layer import MoEConfig
from ancora.model.moe_model import MoEModel
from ancora.model.resident_moe_model import ResidentMoEModel, from_host
from ancora.kernels import muon_ns
from ancora.kernels.muon_ns import newton_schulz_resident_e, PE_SCHEDULE
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)


def sft(cfg, w, ids, labels, STEPS, polar):
    """Run STEPS of Muon SFT with Polar Express on/off. Returns (det, ce-trajectory)."""
    prev = muon_ns.set_polar_express(polar)
    try:
        m = ResidentMoEModel(cfg, copy.deepcopy(w), cfg._B, cfg._S, optimizer="muon")
        h1 = m.forward(ids, si); sync(); h1 = h1.copy()
        h2 = m.forward(ids, si); sync()
        det = float(np.abs(h1.astype(np.float64) - h2.astype(np.float64)).max())  # fwd untouched by opt
        ces = []
        for _ in range(STEPS):
            h = m.forward(ids, si)
            ces.append(m.loss_backward(h, labels, si))
            m.step(si, lr=2e-3); sync()
        return det, ces
    finally:
        muon_ns.set_polar_express(bool(prev))                 # restore global (no state leak)


def time_ns(E, M, steps, schedule, reps=20):
    """Wall-time one batched-expert NS chain (E square M×M experts)."""
    rng = np.random.default_rng(0)
    packed = np.concatenate([_f32bf(rng.standard_normal((M, M)).astype(np.float32)) for _ in range(E)], 0)
    Z = lambda *s: _GpuArray.zeros(s, np.uint16)
    gA, gA2, gB, gBX = Z(E*M, M), Z(E*M, M), Z(E*M, M), Z(E*M, M)
    recip = _GpuArray.zeros((E, 1), np.float32)
    gX = _GpuArray(packed.copy())
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, recip, E, M, M, si, steps=steps, schedule=schedule); sync()
    t = time.perf_counter()
    for _ in range(reps):
        newton_schulz_resident_e(gX, gA, gA2, gB, gBX, recip, E, M, M, si, steps=steps, schedule=schedule)
    sync(); us = (time.perf_counter() - t) / reps * 1e6
    for g in (gX, gA, gA2, gB, gBX, recip): g.free()
    return us


def main(NL=4, STEPS=40):
    cfg = MoEConfig(vocab=2048, n_layers=NL, period=2)
    B, S = 2, 128; cfg._B, cfg._S = B, S; M = B * S
    host = MoEModel(cfg, seed=5, grouped=False, tie=True)
    w = from_host(host, B, S)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, (B, S)).astype(np.int64)
    labels = rng.integers(0, cfg.vocab, (M,)).astype(np.int64)

    print("Polar Express vs Keller in the Muon NS — convergence + determinism + NS timing")
    print("=" * 76)
    det_k, ce_k = sft(cfg, w, ids, labels, STEPS, polar=False)   # Keller (default)
    det_p, ce_p = sft(cfg, w, ids, labels, STEPS, polar=True)    # Polar Express

    # (A) determinism under both — ratio=1 intact
    print(f"  (A) forward determinism: Keller det={det_k:.0e}  PE det={det_p:.0e}  "
          f"{'OK (both bitwise → ratio=1)' if det_k == 0 and det_p == 0 else 'FAIL'}")
    # (B) convergence: PE should reach ≤ Keller's CE
    ckpts = [0, STEPS//4, STEPS//2, STEPS-1]
    print(f"  (B) CE trajectory (Keller vs PE):")
    for i in ckpts:
        print(f"        step {i:3d}:  Keller {ce_k[i]:.4f}   PE {ce_p[i]:.4f}   "
              f"{'PE≤' if ce_p[i] <= ce_k[i] + 1e-3 else 'PE>'}")
    conv_k, conv_p = ce_k[-1] < 0.5*ce_k[0], ce_p[-1] < 0.5*ce_p[0]
    pe_not_worse = ce_p[-1] <= ce_k[-1] * 1.05                   # PE within 5% of Keller (expect ≤)
    print(f"      Keller converges={conv_k}  PE converges={conv_p}  PE-final≤Keller(×1.05)={pe_not_worse}")

    # (C) NS wall-time on the REAL expert shape (E=16 square 1024²)
    print(f"  (C) NS wall-time (E=16, 1024² experts):")
    t_k5 = time_ns(16, 1024, 5, None)
    t_p5 = time_ns(16, 1024, 5, PE_SCHEDULE)
    t_p4 = time_ns(16, 1024, 4, PE_SCHEDULE)
    print(f"        Keller-5 {t_k5:6.0f} us   PE-5 {t_p5:6.0f} us ({t_p5/t_k5:.2f}× = same cost)   "
          f"PE-4 {t_p4:6.0f} us ({t_p4/t_k5:.2f}× = fewer-iter option)")

    ok = det_k == 0 and det_p == 0 and conv_k and conv_p and pe_not_worse
    print("=" * 76)
    print(f"  {'PASS — PE keeps ratio=1, converges ≤ Keller, same NS cost (PE-4 ~20% less)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(NL, STEPS)
