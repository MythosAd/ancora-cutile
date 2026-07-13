"""Timing-only NS fused-vs-unfused (fresh process — the correctness probe heats the GPU and the
laptop clock state makes co-resident timings bimodal; trust consecutive consistent repeats)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.muon_ns import newton_schulz_resident_e
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
def sync(): cudart.cudaStreamSynchronize(si)
Z16 = lambda *s: _GpuArray.zeros(s, np.uint16)


def time_chain(E, M, N, fused, steps=5, reps=30):
    rng = np.random.default_rng(0)
    packed = np.concatenate([_f32bf(rng.standard_normal((M, N)).astype(np.float32)) for _ in range(E)], 0)
    gX = _GpuArray(packed.copy())
    gA, gA2, gB, gBX = Z16(E * M, M), Z16(E * M, M), Z16(E * M, M), Z16(E * M, N)
    rec = _GpuArray.zeros((E, 1), np.float32)
    newton_schulz_resident_e(gX, gA, gA2, gB, gBX, rec, E, M, N, si, steps=steps, schedule=None, fused=fused)
    sync()
    best = 1e9
    for _ in range(3):                                  # best-of-3 windows (clock-state bimodality)
        t = time.perf_counter()
        for _ in range(reps):
            newton_schulz_resident_e(gX, gA, gA2, gB, gBX, rec, E, M, N, si, steps=steps, schedule=None, fused=fused)
        sync(); best = min(best, (time.perf_counter() - t) / reps * 1e6)
    for g in (gX, gA, gA2, gB, gBX, rec): g.free()
    return best


if __name__ == "__main__":
    for E, M, N, tag in ((16, 1024, 1024, "square E=16 1024²   (expert chain)"),
                         (24, 1024, 2048, "rect   E=24 1024x2048 (q/o group)  "),
                         (1, 1024, 1024, "square E=1  1024²   (per-weight)   "),
                         (1, 1024, 2048, "rect   E=1  1024x2048 (per-weight) ")):
        tu = time_chain(E, M, N, fused=False)
        tf = time_chain(E, M, N, fused=True)
        print(f"  {tag}: unfused {tu:6.0f} us → fused {tf:6.0f} us ({tu/tf:.2f}×)")
