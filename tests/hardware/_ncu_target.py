"""Minimal Nsight Compute target: launch a single kernel many times so `ncu --launch-skip N
--launch-count 1` profiles ONE clean steady-state launch (past JIT/warmup). Pick the kernel
via argv: gemm | gateup | attn. Used to ground the MFU claims in real profiler counters
(tensor-core %, DRAM %, occupancy) instead of my own microbench peaks."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
which = sys.argv[1] if len(sys.argv) > 1 else "gemm"

class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.nb = a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))
rng = np.random.default_rng(0)
N_LAUNCH = 12

if which == "gemm":   # MXFP8 GEMM, gate/up shape (M=8192,N=3072,K=1024)
    from ancora.kernels.linear import _fwd_mxfp8_bf16, mxfp8_tile, B as QB
    M, N, K = 8192, 3072, 1024; TM, TN, TK = mxfp8_tile(N, K)
    x = GA(rng.integers(0, 255, (M, K)).astype(np.uint8)); w = GA(rng.integers(0, 255, (K, N)).astype(np.uint8))
    xs = GA(np.full((M, K // QB), 127, np.uint8)); ws = GA(np.full((K // QB, N), 127, np.uint8)); o = GA.z((M, N), np.uint16)
    fn = lambda: ct.launch(SI, (M // TM, N // TN, 1), _fwd_mxfp8_bf16, (x, w, xs, ws, o, K // TK, TM, TN, TK))
elif which == "gateup":
    from ancora.kernels.fused import _gateup_swiglu_q
    from ancora.kernels.quant import B as QB
    M, H, I = 8192, 1024, 3072
    hf = GA(rng.integers(0, 255, (M, H)).astype(np.uint8)); hs = GA(np.full((M, H // QB), 127, np.uint8))
    wg = GA(rng.integers(0, 255, (H, I)).astype(np.uint8)); wgs = GA(np.full((H // QB, I), 127, np.uint8))
    wu = GA(rng.integers(0, 255, (H, I)).astype(np.uint8)); wus = GA(np.full((H // QB, I), 127, np.uint8))
    af = GA.z((M, I), np.uint8); asc = GA.z((M, I // QB), np.uint8)
    fn = lambda: ct.launch(SI, (M // 128, I // 32, 1), _gateup_swiglu_q, (hf, hs, wg, wgs, wu, wus, af, asc, H // 128, 128, 128))
elif which == "gemmdw":   # training weight-grad GEMM (dominant GPU kernel, ~30%) at the M=1024 microbatch
    from ancora.kernels.fused import _gemm_dW
    T = 64; M, K, N = 1024, 1024, 3072   # gate_proj dW: xb (M,H) dy (M,I) → out (H,I)
    bf = lambda a: (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    xb = GA(bf(rng.standard_normal((M, K)).astype(np.float32)))
    dy = GA(bf(rng.standard_normal((M, N)).astype(np.float32)))
    out = GA.z((K, N), np.float32)
    fn = lambda: ct.launch(SI, (K // T, N // T, 1), _gemm_dW, (xb, dy, out, M // T, T, T, T))
elif which == "attn":
    from ancora.kernels.attention import _attn_fwd_tok, BQ, D
    B, S, Hq, Hkv = 4, 2048, 16, 8; M, qd, kd = B * S, Hq * D, Hkv * D; NQB = S // BQ; scale = 1.0 / math.sqrt(D)
    bf = lambda a: (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    Q = GA(bf(rng.standard_normal((M, qd)).astype(np.float32))); Kk = GA(bf(rng.standard_normal((M, kd)).astype(np.float32)))
    Vv = GA(bf(rng.standard_normal((M, kd)).astype(np.float32))); O = GA.z((M, qd), np.uint16); L = GA.z((M * Hq, 1), np.float32)
    fn = lambda: ct.launch(SI, (NQB, B * Hq, 1), _attn_fwd_tok, (Q, Kk, Vv, O, L, NQB, NQB, Hq, Hkv, scale))

for _ in range(N_LAUNCH):
    fn()
cudart.cudaStreamSynchronize(SI)
print(f"done {which}")
