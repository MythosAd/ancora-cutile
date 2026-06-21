"""The 7 Qwen3-layer projections (q/k/v/o/gate/up/down) at the REAL layer size M=B*S=8192,
BF16 vs MXFP8 (amortized on-device quant + ct.mma_scaled). Answers the MFU question for the
FLOP-dominant block: does MXFP8 lift the projections toward 80% of BF16-equivalent peak?

Quant amortized: 4 launches feed 7 GEMMs (gh→q/k/v, attn→o, h2→gate/up, act→down). The quant
cost is counted in the MXFP8 time (honest). Per-shape autotuned tiles. Keep."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.fused import _gemm_bf16
from ancora.kernels.loss import GTM, GTN, GTK
from ancora.kernels.linear import _fwd_mxfp8_bf16, mxfp8_tile
from ancora.kernels.quant import _quant_mxfp8, QTM, B as QB

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
PEAK_BF16, PEAK_MXFP8 = 165.0, 330.0

def _bf(x): return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
class GA:
    def __init__(s, a):
        s.nb = a.nbytes; _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

def time_ms(launch, iters=50, warmup=12):
    for _ in range(warmup): launch()
    so.sync(); _, t0 = cudart.cudaEventCreate(); _, t1 = cudart.cudaEventCreate()
    cudart.cudaEventRecord(t0, SI)
    for _ in range(iters): launch()
    cudart.cudaEventRecord(t1, SI); cudart.cudaEventSynchronize(t1)
    return cudart.cudaEventElapsedTime(t0, t1)[1] / iters


def bench(B=4, S=2048, H=1024, Hq=16, Hkv=8, Dh=64, I=3072):
    M, qd, kd = B * S, Hq * Dh, Hkv * Dh
    rng = np.random.default_rng(0)
    # projection list: (name, input_act, K, N)
    projs = [("q", "gh", H, qd), ("k", "gh", H, kd), ("v", "gh", H, kd), ("o", "go", qd, H),
             ("gate", "g2", H, I), ("up", "g2", H, I), ("down", "ga", I, H)]
    # activation buffers (bf16 bits) + their fp8/scale
    acts = {"gh": H, "go": qd, "g2": H, "ga": I}
    A = {k: GA(_bf(rng.standard_normal((M, w)).astype(np.float32))) for k, w in acts.items()}
    Af = {k: GA.z((M, w), np.uint8) for k, w in acts.items()}
    As = {k: GA.z((M, w // QB), np.uint8) for k, w in acts.items()}
    Wbf = {nm: GA(_bf(rng.standard_normal((K, N)).astype(np.float32))) for nm, _, K, N in projs}
    Wf8 = {nm: GA((rng.integers(0, 255, (K, N))).astype(np.uint8)) for nm, _, K, N in projs}
    Ws8 = {nm: GA(np.full((K // QB, N), 127, np.uint8)) for nm, _, K, N in projs}
    C = {nm: GA.z((M, N), np.uint16) for nm, _, K, N in projs}
    flop = sum(2.0 * M * K * N for _, _, K, N in projs)

    def run_bf16():
        for nm, a, K, N in projs:
            ct.launch(SI, (M // GTM, N // GTN, 1), _gemm_bf16, (A[a], Wbf[nm], C[nm], K // GTK, GTM, GTN, GTK))
    def run_mxfp8():
        for k, w in acts.items():   # 4 amortized quant launches
            ct.launch(SI, (M // QTM, 1, 1), _quant_mxfp8, (A[k], Af[k], As[k], w // QB))
        for nm, a, K, N in projs:
            TM, TN, TK = mxfp8_tile(N, K)
            ct.launch(SI, (M // TM, N // TN, 1), _fwd_mxfp8_bf16, (Af[a], Wf8[nm], As[a], Ws8[nm], C[nm], K // TK, TM, TN, TK))

    t_bf = time_ms(run_bf16); t_mx = time_ms(run_mxfp8)
    f_bf, f_mx = flop / (t_bf / 1e3) / 1e12, flop / (t_mx / 1e3) / 1e12
    print(f"B={B} S={S} M={M}  (7 projections, {flop/1e9:.0f} GFLOP):")
    print(f"  BF16  {t_bf:.3f} ms  {f_bf:6.1f} TFLOPS  = {f_bf/PEAK_BF16*100:.0f}% bf16 peak")
    print(f"  MXFP8 {t_mx:.3f} ms  {f_mx:6.1f} TFLOPS  = {f_mx/PEAK_BF16*100:.0f}% bf16 peak / {f_mx/PEAK_MXFP8*100:.0f}% mxfp8 peak  (incl. quant)")
    print(f"  → MXFP8 speedup {t_bf/t_mx:.2f}x")


if __name__ == "__main__":
    print("Qwen3 projections: BF16 vs MXFP8 at real layer size"); print("=" * 70)
    bench(1, 2048); print("-" * 70)
    bench(4, 2048)
    print("=" * 70)
