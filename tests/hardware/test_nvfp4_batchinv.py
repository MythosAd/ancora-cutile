"""STEP 4: batch-invariance + determinism of the CUTLASS NVFP4 GEMM (RL-critical), plus the
callable perf vs MXFP8. Determinism: same input twice → bitwise identical. Batch-invariance:
output row m identical for M=256 vs M=512 (same A-row + same B → same logits, no split-K)."""
import sys, os, ctypes, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.quant_nvfp4 import quantize_nvfp4_rowblock

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])

def _bind(name):
    d = ctypes.CDLL(name)
    pre = "cutlass_nvfp4" if "nvfp4" in name else "cutlass_mxfp8"
    getattr(d, pre + "_init").argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3; getattr(d, pre + "_init").restype = ctypes.c_void_p
    getattr(d, pre + "_set_scales").argtypes = [ctypes.c_void_p] * 4
    getattr(d, pre + "_run").argtypes = [ctypes.c_void_p] * 2; getattr(d, pre + "_run").restype = ctypes.c_int
    getattr(d, pre + "_free").argtypes = [ctypes.c_void_p]
    return d
NV = _bind(r"C:\project\cutlass\cutlass_nvfp4.dll")
def dput(a): a = np.ascontiguousarray(a); _, p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(p, a, a.nbytes); return p
def dget(p, sh, dt): o = np.empty(sh, dt); cdrv.cuMemcpyDtoH(o, p, o.nbytes); return o
cv = lambda p: ctypes.c_void_p(int(p))


def run_nvfp4(M, N, K, fpA, sfA, fpBt, sfBt):
    dA, dB, dD = dput(fpA), dput(fpBt), cdrv.cuMemAlloc(M * N * 2)[1]
    dSFA, dSFB = dput(sfA), dput(sfBt)
    h = NV.cutlass_nvfp4_init(M, N, K, cv(dA), cv(dB), cv(dD))
    NV.cutlass_nvfp4_set_scales(h, cv(dSFA), cv(dSFB), cv(SI)); NV.cutlass_nvfp4_run(h, cv(SI))
    cudart.cudaStreamSynchronize(SI)
    out = dget(dD, (M, N), np.uint16).copy()
    return h, out, (dA, dB, dD, dSFA, dSFB)


if __name__ == "__main__":
    print("NVFP4 GEMM — batch-invariance + determinism (step 4)"); print("=" * 64)
    N, K = 1024, 1024; rng = np.random.default_rng(0)
    Bm = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    fpBt, sfBt = quantize_nvfp4_rowblock(Bm.T.copy())
    # build A for M=512; M=256 reuses the FIRST 256 rows (so row m's data is identical across batch sizes)
    A512 = (rng.standard_normal((512, K)) * 0.5).astype(np.float32)
    fA512, sA512 = quantize_nvfp4_rowblock(A512)
    fA256, sA256 = quantize_nvfp4_rowblock(A512[:256].copy())

    h2, out512, bufs2 = run_nvfp4(512, N, K, fA512, sA512, fpBt, sfBt)
    h1, out256, bufs1 = run_nvfp4(256, N, K, fA256, sA256, fpBt, sfBt)
    # (a) determinism: re-run M=512, bitwise identical
    NV.cutlass_nvfp4_run(h2, cv(SI)); cudart.cudaStreamSynchronize(SI)
    out512b = dget(bufs2[2], (512, N), np.uint16)
    det = np.array_equal(out512, out512b)
    print(f"  (a) determinism (same input twice bitwise ==): {'OK' if det else 'FAIL'}")
    # (b) batch invariance: row 0..255 identical for M=256 vs M=512
    binv = np.array_equal(out256, out512[:256])
    print(f"  (b) batch-size invariance (rows 0..255: M=256 == M=512 prefix): {'OK — bitwise' if binv else 'FAIL'}")

    # perf vs MXFP8 (callable), layer shapes
    try:
        MX = _bind(r"C:\project\cutlass\cutlass_mxfp8.dll")
        def tms(fn, it=40, wm=10):
            for _ in range(wm): fn()
            cudart.cudaStreamSynchronize(SI); t = time.perf_counter()
            for _ in range(it): fn()
            cudart.cudaStreamSynchronize(SI); return (time.perf_counter() - t) / it * 1e3
        print("  (c) callable perf NVFP4 vs MXFP8 (layer shapes):")
        for (M, Np, Kp, lab) in [(8192, 3072, 1024, "gate/up"), (8192, 1024, 3072, "down"), (8192, 1024, 1024, "q/o")]:
            rngp = np.random.default_rng(1)
            fa, sa = quantize_nvfp4_rowblock((rngp.standard_normal((M, Kp)) * 0.5).astype(np.float32))
            fb, sb = quantize_nvfp4_rowblock((rngp.standard_normal((Np, Kp)) * 0.3).astype(np.float32))
            hn, _, bn = run_nvfp4(M, Np, Kp, fa, sa, fb, sb)
            tn = tms(lambda: NV.cutlass_nvfp4_run(hn, cv(SI)))
            # mxfp8 callable (random fp8 bytes — perf only)
            mA = dput(rngp.integers(0, 255, (M, Kp)).astype(np.uint8)); mB = dput(rngp.integers(0, 255, (Kp, Np)).astype(np.uint8))
            mSA = dput(np.full((M, Kp // 32), 0x70, np.uint8)); mSB = dput(np.full((Np, Kp // 32), 0x70, np.uint8)); mD = cdrv.cuMemAlloc(M * Np * 2)[1]
            hm = MX.cutlass_mxfp8_init(M, Np, Kp, cv(mA), cv(mB), cv(mD)); MX.cutlass_mxfp8_set_scales(hm, cv(mSA), cv(mSB), cv(SI))
            tm = tms(lambda: MX.cutlass_mxfp8_run(hm, cv(SI)))
            fl = 2.0 * M * Np * Kp
            print(f"      {lab:8s} NVFP4 {fl/(tn/1e3)/1e12:.0f} TF | MXFP8 {fl/(tm/1e3)/1e12:.0f} TF | {tm/tn:.2f}x")
    except Exception as e:
        print(f"  (c) perf compare skipped: {str(e)[:80]}")
    print("=" * 64); print(f"  {'PASS' if det and binv else 'FAIL'}")
