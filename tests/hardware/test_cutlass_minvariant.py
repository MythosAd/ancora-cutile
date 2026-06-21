"""GATE TEST for putting CUTLASS MXFP8 GEMM in the BITWISE (ratio=1) path.

ratio π_train/π_infer = 1 needs: a given activation row produces a BITWISE-identical output
row regardless of total M (rollout decode pads M→128; training prefill is M=S). That holds iff
the CUTLASS tile scheduler is DATA-PARALLEL — each output tile owns its full K-reduction in one
CTA, no split-K / no stream-K (whose reduction order would vary with the problem's tile count).
Determinism (same call twice) is necessary but NOT sufficient; this checks M-invariance directly.

  Take a fixed A (Mbig, K). Run CUTLASS at M=Mbig and at M=128 on A's FIRST 128 rows (same data,
  same tile-0 positions). If D_small[r] == D_big[r] BITWISE for all r<128 and all in-tile positions,
  the GEMM is M-invariant → safe for the bitwise path. Also checks an OFFSET frontier row (the decode
  frontier sits at row pmod of its tile, not always row 0).
"""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.quant import quantize_rowblock, quantize_colblock

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])

DLL = ctypes.CDLL(r"C:\project\cutlass\cutlass_mxfp8.dll")
DLL.cutlass_mxfp8_init.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3; DLL.cutlass_mxfp8_init.restype = ctypes.c_void_p
DLL.cutlass_mxfp8_set_scales.argtypes = [ctypes.c_void_p] * 4
DLL.cutlass_mxfp8_run.argtypes = [ctypes.c_void_p] * 2; DLL.cutlass_mxfp8_run.restype = ctypes.c_int
DLL.cutlass_mxfp8_free.argtypes = [ctypes.c_void_p]
cv = lambda p: ctypes.c_void_p(int(p))

def dput(a):
    a = np.ascontiguousarray(a); _, p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(p, a, a.nbytes); return p
def dget_u16(p, shape):
    o = np.empty(shape, np.uint16); cdrv.cuMemcpyDtoH(o, p, o.nbytes); return o

def run_cutlass(M, N, K, fpA, sfA, fpB_cm, sfB_t):
    """Returns the raw (M,N) bf16-bit output (uint16) so comparison is bitwise."""
    dA = dput(fpA); dB = dput(fpB_cm); _, dD = cdrv.cuMemAlloc(M * N * 2)
    dSFA = dput(sfA); dSFB = dput(sfB_t)
    h = DLL.cutlass_mxfp8_init(M, N, K, cv(dA), cv(dB), cv(dD)); assert h, "init failed"
    DLL.cutlass_mxfp8_set_scales(h, cv(dSFA), cv(dSFB), cv(SI))
    DLL.cutlass_mxfp8_run(h, cv(SI)); cudart.cudaStreamSynchronize(SI)
    out = dget_u16(dD, (M, N))
    DLL.cutlass_mxfp8_free(h)
    for p in (dA, dB, dD, dSFA, dSFB): cdrv.cuMemFree(p)
    return out


def test(Mbig, N, K):
    rng = np.random.default_rng(0)
    A  = (rng.standard_normal((Mbig, K)) * 0.5).astype(np.float32)
    Bm = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    fpA, sfA = quantize_rowblock(A)
    fpB, wsc = quantize_colblock(Bm)
    fpB_cm = np.ascontiguousarray(fpB.T)      # (N,K) row-major == (K,N) col-major (CUTLASS B)
    sfB_t  = np.ascontiguousarray(wsc.T)      # (N, K//32)

    D_big   = run_cutlass(Mbig, N, K, fpA, sfA, fpB_cm, sfB_t)
    D_small = run_cutlass(128,  N, K, fpA[:128], sfA[:128], fpB_cm, sfB_t)

    # (a) M-count invariance + in-tile-position invariance: rows 0..127 must match bitwise
    same = np.array_equal(D_small, D_big[:128])
    nbad = int((D_small != D_big[:128]).any(axis=1).sum())
    print(f"  M={Mbig} N={N} K={K}:  D(M=128)[0:128] vs D(M={Mbig})[0:128]  "
          f"{'BITWISE IDENTICAL' if same else f'DIFFER ({nbad}/128 rows)'}  {'OK' if same else 'FAIL'}")

    # (b) offset frontier: the decode frontier can sit at row pmod (not 0). Build M=128 with the
    #     frontier A-row placed at row pmod, compare to the same A-row at row pmod of D_big's tile 0.
    ok_off = True
    for pmod in (0, 1, 50, 127):
        Asmall = fpA[:128].copy(); sAsmall = sfA[:128].copy()
        Asmall[pmod] = fpA[pmod]; sAsmall[pmod] = sfA[pmod]    # frontier row = A[pmod] (already is)
        Ds = run_cutlass(128, N, K, Asmall, sAsmall, fpB_cm, sfB_t)
        m = np.array_equal(Ds[pmod], D_big[pmod]); ok_off &= m
    print(f"           offset-frontier rows {{0,1,50,127}} match D(M={Mbig}) bitwise: {'OK' if ok_off else 'FAIL'}")
    return same and ok_off


if __name__ == "__main__":
    print("CUTLASS MXFP8 M-INVARIANCE gate (is it safe for the bitwise ratio=1 path?)")
    print("=" * 74)
    ok = True
    ok &= test(256,  1024, 1024)    # H→qd-ish
    ok &= test(512,  3072, 1024)    # gate/up shape
    ok &= test(512,  1024, 3072)    # down shape
    print("=" * 74)
    print(f"  {'PASS — CUTLASS GEMM is M-invariant ⇒ can serve the bitwise path' if ok else 'FAIL — NOT M-invariant (split-K?) ⇒ NOT safe for ratio=1'}")
    sys.exit(0 if ok else 1)
