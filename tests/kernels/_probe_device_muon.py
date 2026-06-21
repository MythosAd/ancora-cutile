"""Validate DeviceMuon (ancora.optim.muon.DeviceMuon) vs the validated host fp32 Muon.

(1) step-reproduction: feed the SAME gradient sequence to both, compare the accumulated
    orthogonalized UPDATE (W0 - master) by cosine + relative norm. bf16-on-GPU NS vs fp32
    host NS → ~1-3% norm, cosine ~0.99. Covers square (no transpose), wide (K<N), and TALL
    (K>N → the transpose path, where a scratch-aliasing bug would make the result GARBAGE).
(2) convergence: minimize 0.5||W-T||² (g = W-T) and confirm the device optimizer descends.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
from cuda.bindings import runtime as cudart
from cuda.bindings import driver as cdrv
import ancora.env  # noqa: F401
from ancora.optim.muon import DeviceMuon, Muon, MuonScratch, ResidentMuon
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf
from ancora.model.resident import _DBuf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _free(dm):
    for b in (dm.p32, dm.buf, dm.p16, dm.u, dm.gA, dm.gA2, dm.gB, dm.scr, dm.recip):
        b.free()
    if dm.gBX is not None:
        dm.gBX.free()


def reproduce(K, N, T=5, lr=0.02, seed=0):
    """Same W0 + same grads → compare the total update W0-master after T steps."""
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((K, N)).astype(np.float32) * 0.05
    grads = [rng.standard_normal((K, N)).astype(np.float32) for _ in range(T)]

    host = Muon({"w": W0.copy()}, lr=lr, momentum=0.95, nesterov=True, ns_steps=5)
    for g in grads:
        host.step({"w": g})
    Wh = host.master()["w"]

    dm = DeviceMuon(W0.copy(), lr=lr, momentum=0.95, ns_steps=5)
    g_dev = _GpuArray.zeros((K, N), np.float32)
    for g in grads:
        gc = np.ascontiguousarray(g)
        cdrv.cuMemcpyHtoDAsync(g_dev._ptr, gc, gc.nbytes, si)
        dm.step(g_dev, si)
        cudart.cudaStreamSynchronize(si)
    Wd = dm.master()

    du_h, du_d = W0 - Wh, W0 - Wd            # the accumulated orthogonalized update
    c, r = cos(du_d, du_h), rel(du_d, du_h)
    ok = c > 0.99 and r < 0.06
    tag = "tall(transpose)" if K > N else ("wide" if K < N else "square")
    print(f"  ({K},{N}) {tag:16s}: update cos {c:.4f}  rel {r:.2%}  {'OK' if ok else 'FAIL'}")
    g_dev.free(); _free(dm)
    return ok


def converge(K, N, steps=30, lr=0.02, seed=1):
    """Minimize 0.5||W-T||²: g = W-T. Device must descend; host printed for reference."""
    rng = np.random.default_rng(seed)
    Tgt = rng.standard_normal((K, N)).astype(np.float32) * 0.02   # ‖T‖_F ≈ 20 → reachable in ~30 steps
    W0 = np.zeros((K, N), np.float32)

    host = Muon({"w": W0.copy()}, lr=lr, momentum=0.95, ns_steps=5)
    lh = []
    for _ in range(steps):
        g = host.master()["w"] - Tgt
        lh.append(0.5 * float(np.sum(g * g)))
        host.step({"w": g})

    dm = DeviceMuon(W0.copy(), lr=lr, momentum=0.95, ns_steps=5)
    g_dev = _GpuArray.zeros((K, N), np.float32)
    ld = []
    for _ in range(steps):
        g = dm.master() - Tgt
        ld.append(0.5 * float(np.sum(g * g)))
        gc = np.ascontiguousarray(g)
        cdrv.cuMemcpyHtoDAsync(g_dev._ptr, gc, gc.nbytes, si)
        dm.step(g_dev, si)
        cudart.cudaStreamSynchronize(si)
    drop_d, drop_h = ld[-1] / ld[0], lh[-1] / lh[0]
    ok = drop_d < 0.2 and drop_d < 3 * drop_h + 0.05     # clear descent + not wildly off host
    tag = "tall" if K > N else "square"
    print(f"  ({K},{N}) {tag:6s} converge: device {ld[0]:.0f}->{ld[-1]:.1f} ({drop_d:.2%})  "
          f"host {lh[0]:.0f}->{lh[-1]:.1f} ({drop_h:.2%})  {'OK' if ok else 'FAIL'}")
    g_dev.free(); _free(dm)
    return ok


def reproduce_resident(T=5, lr=0.02, seed=0):
    """ResidentMuon over EXISTING p32/p16 buffers + ONE SHARED MuonScratch across square/wide/tall
    weights, all stepping in the same steps. Proves (a) shared scratch doesn't cross-contaminate,
    (b) the tall transpose path is correct, (c) it matches host Muon — the integration vehicle."""
    shapes = [(1024, 1024), (1024, 2048), (2048, 1024)]   # square, wide, TALL — share ONE scratch
    rng = np.random.default_rng(seed)
    sc = MuonScratch(shapes)
    print(f"  shared scratch: {sc.nbytes() / 1e6:.1f} MB (one-time, all weights)")
    rms = []
    for K, N in shapes:
        W0 = rng.standard_normal((K, N)).astype(np.float32) * 0.05
        host = Muon({"w": W0.copy()}, lr=lr, momentum=0.95, ns_steps=5)
        p32 = _DBuf(W0.astype(np.float32).copy()); p16 = _DBuf(_f32bf(W0))
        rm = ResidentMuon(p32, p16, K, N, sc, momentum=0.95, ns_steps=5)
        g_dev = _DBuf.zeros((K, N), np.float32)
        rms.append((K, N, W0, host, p32, p16, rm, g_dev))
    ok = True
    for t in range(T):                                     # interleave: all weights share sc each step
        for K, N, W0, host, p32, p16, rm, g_dev in rms:
            g = rng.standard_normal((K, N)).astype(np.float32)
            host.step({"w": g})
            gc = np.ascontiguousarray(g)
            cdrv.cuMemcpyHtoDAsync(g_dev.ptr, gc, gc.nbytes, si)
            rm.step(g_dev, si, lr)
    cudart.cudaStreamSynchronize(si)
    for K, N, W0, host, p32, p16, rm, g_dev in rms:
        Wd = p32.to_numpy(); Wh = host.master()["w"]
        c, r = cos(W0 - Wd, W0 - Wh), rel(W0 - Wd, W0 - Wh)
        tag = "tall(transpose)" if K > N else ("wide" if K < N else "square")
        good = c > 0.99 and r < 0.06
        print(f"  ({K},{N}) {tag:16s}: update cos {c:.4f}  rel {r:.2%}  {'OK' if good else 'FAIL'}")
        ok &= good
        p32.free(); p16.free(); rm.free(); g_dev.free()
    sc.free()
    return ok


if __name__ == "__main__":
    ok = True
    print("step-reproduction (device bf16 NS == host fp32 NS within tolerance):")
    ok &= reproduce(1024, 1024)     # square (down_proj), no transpose
    ok &= reproduce(1024, 2048)     # wide (q/k/v_proj H×qd), no transpose
    ok &= reproduce(2048, 1024)     # TALL (o_proj qd×H), K>N → transpose path (the scratch fix)
    print("convergence (descends like host):")
    ok &= converge(1024, 1024)
    ok &= converge(2048, 1024)      # tall convergence
    print("resident shared-scratch (the integration vehicle):")
    ok &= reproduce_resident()
    print("  PASS" if ok else "  FAIL")
