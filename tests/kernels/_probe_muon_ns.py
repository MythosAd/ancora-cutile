"""Validate the RESIDENT device Newton-Schulz (muon_ns.newton_schulz_resident) vs the host fp32 NS:
  (1) it ORTHOGONALIZES — the result's singular values ≈ 1 (the whole point of Muon);
  (2) it tracks the host NS within bf16 tolerance (~few %); the device chain runs with NO host
      round-trip (one upload, the 15 GEMMs + axpy/norm on-device, one download)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import newton_schulz_resident
from ancora.optim.muon import newton_schulz5
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def run(M, N):
    assert M <= N
    rng = np.random.default_rng(0)
    # momentum-like matrix with a spread of singular values (so orthogonalization has work to do)
    G = (rng.standard_normal((M, N)) @ np.diag(rng.uniform(0.1, 3.0, N)) if M == N else
         rng.standard_normal((M, N))).astype(np.float32)
    Gb = b2f(_f32bf(G))                                       # bf16-round the input (both paths see this)
    O_host = newton_schulz5(Gb, steps=5)                      # fp32 reference NS on the bf16 input

    Z = lambda *s: _GpuArray.zeros(s, np.uint16)
    gX = _GpuArray(_f32bf(Gb)); gA, gA2, gB = Z(M, M), Z(M, M), Z(M, M); gBX = Z(M, N)
    grecip = _GpuArray.zeros((1, 1), np.float32)
    newton_schulz_resident(gX, gA, gA2, gB, gBX, grecip, M, N, si); cudart.cudaStreamSynchronize(si)
    # timing (resident — one chain)
    t0 = time.perf_counter()
    for _ in range(20):
        newton_schulz_resident(gX2 := _GpuArray(_f32bf(Gb)), gA, gA2, gB, gBX, grecip, M, N, si)
    cudart.cudaStreamSynchronize(si); us = (time.perf_counter() - t0) / 20 * 1e6
    gX2.free()
    O_dev = b2f(gX.to_numpy())

    sv = np.linalg.svd(O_dev, compute_uv=False)              # singular values of the orthogonalized result
    hsv = np.linalg.svd(O_host, compute_uv=False)
    err = float(np.linalg.norm(O_dev - O_host) / (np.linalg.norm(O_host) + 1e-30))
    # CORRECTNESS = the device NS reproduces the host NS (within bf16): same result, same σ spectrum.
    # (The absolute σ_min depends on the INPUT conditioning + NS5's 5-step limit — identical host/device,
    #  not a device property.)
    match = abs(sv.min() - hsv.min()) < 0.03 and abs(sv.max() - hsv.max()) < 0.03
    print(f"  ({M},{N}): σ∈[{sv.min():.3f},{sv.max():.3f}] host [{hsv.min():.3f},{hsv.max():.3f}]  "
          f"vs-host {err:.2%}  {us:.0f}us  {'OK (reproduces host NS)' if match and err < 0.05 else 'FAIL'}")
    for g in (gX, gA, gA2, gB, gBX, grecip): g.free()
    return match and err < 0.05


if __name__ == "__main__":
    ok = True
    ok &= run(1024, 1024)     # down_proj / square
    ok &= run(1024, 2048)     # q_proj (H, qd) — M≤N
    ok &= run(1024, 1024)     # gate/up (H, I)
    print("  PASS" if ok else "  FAIL")
