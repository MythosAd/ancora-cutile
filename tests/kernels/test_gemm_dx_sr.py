"""_gemm_dx_sr: stochastic-rounding variant of the activation-gradient GEMM (dy@Wᵀ).
Verifies (a) it compiles + runs, (b) the SR downcast is UNBIASED — averaging the bf16 SR output
over many seeds recovers the f32 result, whereas plain _gemm_dx (round-to-nearest) is a single
fixed bf16 rounding. (c) determinism for a FIXED seed (batch-invariance: same seed → same bits)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
import ancora.env  # noqa
from ancora.kernels.fused import _gemm_dx, _gemm_dx_sr

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

class G:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o

f32bf = lambda x: (x.astype(np.float32).view(np.uint32) + 0x7FFF + ((x.astype(np.float32).view(np.uint32) >> 16) & 1) >> 16).astype(np.uint16)
bf32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
_bf = lambda x: bf32(f32bf(x))
rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

M, K, N = 64, 128, 128
T = 64
rng = np.random.default_rng(0)
dy = (rng.standard_normal((M, N)) * 0.5).astype(np.float32)
W = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
ref = _bf(dy).astype(np.float64) @ _bf(W).astype(np.float64).T     # (M,K) f32 reference (bf16 inputs)

gdy = G(f32bf(dy)); gW = G(f32bf(W))


def run_rtn():
    gdx = G(np.zeros((M, K), np.uint16))
    ct.launch(si, (M // T, K // T, 1), _gemm_dx, (gdy, gW, gdx, N // T, T, T, T))
    cudart.cudaStreamSynchronize(si)
    return bf32(gdx.np())


def run_sr(seed):
    gdx = G(np.zeros((M, K), np.uint16))
    ct.launch(si, (M // T, K // T, 1), _gemm_dx_sr, (gdy, gW, gdx, N // T, T, T, T, int(seed)))
    cudart.cudaStreamSynchronize(si)
    return bf32(gdx.np())


print("--- _gemm_dx_sr: compiles, unbiased, fixed-seed deterministic ---")
rtn = run_rtn()
print(f"  RTN  (_gemm_dx)        rel vs f32: {rel(rtn, ref)*100:.3f}%   (single bf16 rounding)")

NS = 600
acc = np.zeros((M, K), np.float64)
for s in range(NS):
    acc += run_sr(s)
sr_mean = acc / NS
print(f"  SR mean over {NS} seeds  rel vs f32: {rel(sr_mean, ref)*100:.3f}%   (should be << RTN — unbiased)")

# fixed-seed determinism (batch-invariance requirement)
a1, a2 = run_sr(42), run_sr(42)
same = np.array_equal(f32bf(a1), f32bf(a2))
# and a single SR draw is bf16-quantized (coarser than the mean)
single_err = rel(run_sr(1), ref)
print(f"  fixed seed=42 twice → bitwise identical: {same}")
print(f"  single SR draw rel vs f32: {single_err*100:.3f}%  (bf16-coarse per draw, unbiased in mean)")

ok = rel(sr_mean, ref) < rel(rtn, ref) * 0.5 and same
print("=" * 56)
print(f"  {'PASS — SR unbiased (mean beats RTN) + seed-deterministic' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
