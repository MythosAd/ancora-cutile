"""Unit-test _gemm_dx_fp8 (the E4M3 dgrad kernel) at real backward shapes + tile T=64:
  dx = dy @ Wᵀ.  Compare to (a) the fp64 reference and (b) the BF16 _gemm_dx (the current path).
Quantizes dy + W per-32 along N with _quant_mxfp8 (the existing along-axis-1 quant), exactly as the
resident layer will. Confirms the kernel is correct and that FP8 dgrad error ≈ the ~3.8% E4M3 floor
(vs BF16's near-exact), at the layer's real (K,N) shapes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.fused import _gemm_dx, _gemm_dx_fp8
from ancora.kernels.quant import _quant_mxfp8, QTM, B as QB
from ancora.kernels.loss import _GpuArray, f32_to_bf16_bits as _f32bf

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])
T = 64


def run(name, M, K, N):
    rng = np.random.default_rng(0)
    # dy = gradient (wide per-row dynamic range, like a real activation grad); W = weight (K,N)
    mag = 10.0 ** rng.uniform(-2.5, 0.5, size=(M, 1))
    dy = (rng.standard_normal((M, N)).astype(np.float32) * mag).astype(np.float32)
    W = (rng.standard_normal((K, N)).astype(np.float32) * 0.5)
    dyb, Wb = _f32bf(dy), _f32bf(W)                                  # bf16 bits (what the layer holds)
    ref = (dy.astype(np.float64).view(np.float64)) @ W.astype(np.float64).T  # but use bf16-rounded operands:
    dyr = (dyb.astype(np.uint32) << 16).view(np.float32); Wr = (Wb.astype(np.uint32) << 16).view(np.float32)
    ref = dyr.astype(np.float64) @ Wr.astype(np.float64).T; rn = np.abs(ref).max() + 1e-30
    gdy, gW = _GpuArray(dyb), _GpuArray(Wb)

    # BF16 _gemm_dx (current path)
    dxb = _GpuArray(np.zeros((M, K), np.uint16))
    ct.launch(si, (M // T, K // T, 1), _gemm_dx, (gdy, gW, dxb, N // T, T, T, T))
    cudart.cudaStreamSynchronize(si)
    e_bf = np.abs((dxb.to_numpy().astype(np.uint32) << 16).view(np.float32).astype(np.float64) - ref).max() / rn

    # FP8 E4M3 dgrad: quant dy + W per-32 along N, then _gemm_dx_fp8
    dyf, dys = _GpuArray(np.zeros((M, N), np.uint8)), _GpuArray(np.zeros((M, N // QB), np.uint8))
    Wf, Ws = _GpuArray(np.zeros((K, N), np.uint8)), _GpuArray(np.zeros((K, N // QB), np.uint8))
    dxf = _GpuArray(np.zeros((M, K), np.uint16))
    ct.launch(si, (M // QTM, 1, 1), _quant_mxfp8, (gdy, dyf, dys, N // QB))
    ct.launch(si, (K // QTM, 1, 1), _quant_mxfp8, (gW, Wf, Ws, N // QB))
    ct.launch(si, (M // T, K // T, 1), _gemm_dx_fp8, (dyf, dys, Wf, Ws, dxf, N // T, T, T, T))
    cudart.cudaStreamSynchronize(si)
    e_fp8 = np.abs((dxf.to_numpy().astype(np.uint32) << 16).view(np.float32).astype(np.float64) - ref).max() / rn
    print(f"  {name:22s} (M={M},K={K},N={N}): BF16 dgrad {e_bf:.3%}  |  FP8-E4M3 dgrad {e_fp8:.3%}  "
          f"{'OK' if e_fp8 < 0.07 else 'FAIL'}")
    return e_fp8 < 0.07


if __name__ == "__main__":
    # real Qwen3/MoE projection shapes (the dgrad N = the proj output dim)
    H, qd, kd, I, M = 1024, 2048, 1024, 1024, 256
    ok = True
    ok &= run("q_proj dx", M, H, qd)      # dx = dy(M,qd) @ q_projᵀ(qd,H) → (M,H); reduce N=qd
    ok &= run("o_proj dx", M, qd, H)      # reduce N=H
    ok &= run("gate_proj dx", M, H, I)    # reduce N=I
    ok &= run("down_proj dx", M, I, H)    # reduce N=H
    print("  PASS" if ok else "  FAIL")
