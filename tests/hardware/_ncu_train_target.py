"""ncu target for the TRAINING-step hot kernels (M=1024 real shapes). Launches one kernel 12×
so --launch-skip 8 --launch-count 1 profiles a steady-state launch.
Usage: _ncu_train_target.py {dw|dhid|proj|cegrad|adamw}"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import cuda.core as cc
import cuda.tile as ct
from cuda.bindings import runtime as cudart
import ancora.env  # noqa: F401
from ancora.kernels.fused import _gemm_dW, _gemm_bf16
from ancora.kernels.loss import _gemm, _ce_grad, _GpuArray, f32_to_bf16_bits, GTM, GTN, GTK, CTM, TV
from ancora.optim.adamw import _adamw, _pick_otm, C as AC

cudart.cudaFree(0)
dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); si = int(so.__cuda_stream__()[1])

M, H, V = 1024, 1024, 151936
rng = np.random.default_rng(0)
G = lambda a: _GpuArray(np.ascontiguousarray(a))
BF = lambda *s: G(f32_to_bf16_bits(rng.standard_normal(s, np.float32)))

mode = sys.argv[1]
if mode == "dw":            # boundary LM-head dW: (V,M)@(M,H) → (V,H) f32, T=64
    T = 64
    gg, gh, dW = BF(M, V), BF(M, H), G(np.zeros((V, H), np.float32))
    fn = lambda: ct.launch(si, (V // T, H // T, 1), _gemm_dW, (gg, gh, dW, M // T, T, T, T))
elif mode == "dhid":        # dhidden: (M,V)@(V,H) → (M,H), GTM/GTN/GTK
    gg, ge, dh = BF(M, V), BF(V, H), G(np.zeros((M, H), np.float32))
    fn = lambda: ct.launch(si, (M // GTM, H // GTN, 1), _gemm, (gg, ge, dh, V // GTK, GTM, GTN, GTK))
elif mode == "proj":        # layer projection fwd: (M,H)@(H,H) bf16, 128/128/64 (the 64-block case)
    A, B, C = BF(M, H), BF(H, H), G(np.zeros((M, H), np.uint16))
    fn = lambda: ct.launch(si, (M // 128, H // 128, 1), _gemm_bf16, (A, B, C, H // 64, 128, 128, 64))
elif mode == "cegrad":      # CE grad stream: (M,V) f32 → bf16, CTM=64 → 16 blocks
    lg, lse = G(rng.standard_normal((M, V), np.float32)), G(np.zeros((M, 1), np.float32))
    lab = G(rng.integers(0, V, (M, 1)).astype(np.int32)); adv = G(np.ones((M, 1), np.float32))
    out = G(np.zeros((M, V), np.uint16))
    fn = lambda: ct.launch(si, (M // CTM, 1, 1), _ce_grad, (lg, lse, lab, adv, out, V // TV, 1.0 / M))
elif mode == "adamw":       # expert-weight AdamW: R=16.8M/AC rows
    R = (16 * H * 1024) // AC; otm = _pick_otm(R)
    g, m, v = G(np.zeros((R, AC), np.float32)), G(np.zeros((R, AC), np.float32)), G(np.zeros((R, AC), np.float32))
    p32, p16 = G(np.zeros((R, AC), np.float32)), G(np.zeros((R, AC), np.uint16))
    fn = lambda: ct.launch(si, (R // otm, 1, 1), _adamw, (g, m, v, p32, p16, otm,
                           0.9, 0.999, 1e-8, 1e-4, 0.0, 1.0, 1.0))
for _ in range(12):
    fn()
cudart.cudaStreamSynchronize(si)
print(f"{mode}: done")
