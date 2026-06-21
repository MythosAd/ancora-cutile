"""Validate the BATCHED-expert resident Newton-Schulz (muon_ns.newton_schulz_resident_e) vs a
per-expert host NS. The batched kernels put the expert dim in the grid (bid(0)) with row-block
offset e*(M//T) — the key correctness risk is CROSS-EXPERT contamination (a wrong offset mixing
experts). So: pack E DIFFERENT random matrices, run ONE batched NS, and check EACH expert's result
matches its OWN host NS (a leak would make some experts wrong while the aggregate looks fine)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import newton_schulz_resident_e
from ancora.optim.muon import newton_schulz5
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def run(E, M):
    N = M                                                   # uniform-square experts
    rng = np.random.default_rng(0)
    # E DISTINCT experts, each with a spread of singular values (so NS has real work + a leak shows)
    Ws = [(rng.standard_normal((M, N)) @ np.diag(rng.uniform(0.1, 3.0, N))).astype(np.float32) for _ in range(E)]
    packed = np.concatenate([_f32bf(b2f(_f32bf(W))) for W in Ws], axis=0)   # (E*M, N) bf16 bits
    host = [newton_schulz5(b2f(_f32bf(W)), steps=5) for W in Ws]            # per-expert host NS on the bf16 input

    Z = lambda *s: _GpuArray.zeros(s, np.uint16)
    gX = _GpuArray(packed); gA, gA2, gB, gBX = Z(E * M, M), Z(E * M, M), Z(E * M, M), Z(E * M, N)
    grecip = _GpuArray.zeros((E, 1), np.float32)
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, grecip, E, M, N, si); cudart.cudaStreamSynchronize(si)
    # timing
    t0 = time.perf_counter()
    for _ in range(20):
        newton_schulz_resident_e(gX2 := _GpuArray(packed), gA, gA2, gB, gBX, grecip, E, M, N, si)
    cudart.cudaStreamSynchronize(si); us = (time.perf_counter() - t0) / 20 * 1e6; gX2.free()
    out = b2f(gX.to_numpy())                                # (E*M, N)

    ok = True
    worst_c, worst_r = 1.0, 0.0
    for e in range(E):
        Od = out[e * M:(e + 1) * M]; Oh = host[e]
        c = float((Od.ravel() @ Oh.ravel()) / (np.linalg.norm(Od) * np.linalg.norm(Oh) + 1e-30))
        r = float(np.linalg.norm(Od - Oh) / (np.linalg.norm(Oh) + 1e-30))
        worst_c, worst_r = min(worst_c, c), max(worst_r, r)
        ok &= (c > 0.99 and r < 0.05)
    print(f"  E={E} M={M}: worst expert cos {worst_c:.4f}  rel {worst_r:.2%}  {us:.0f}us/chain  "
          f"{'OK (no cross-expert leak)' if ok else 'FAIL'}")
    for g in (gX, gA, gA2, gB, gBX, grecip): g.free()
    return ok


if __name__ == "__main__":
    ok = True
    ok &= run(4, 256)       # small, fast — checks the indexing
    ok &= run(16, 1024)     # the REAL MoE shape (E=16 square 1024×1024 experts)
    print("  PASS" if ok else "  FAIL")
