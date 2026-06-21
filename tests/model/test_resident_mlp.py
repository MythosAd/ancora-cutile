"""Device-resident MLP block (rmsnorm→gate/up GEMM→SwiGLU→down GEMM→residual), chained
on persistent buffers, vs the host-glue MLP. Validates correctness (bf16) AND the end-to-end
speedup that device-residency buys. First component of the device-resident forward."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import runtime as cudart

from ancora.kernels.norm import (_rmsnorm_stats, _rmsnorm_apply, rmsnorm_forward,
                                  TM as NTM, TH, _GpuArray as GA, f32_to_bf16_bits as f32bf,
                                  bf16_bits_to_f32 as bf32)
from ancora.kernels.activation import _swiglu_fwd, swiglu_forward, TM as STM, TI
from ancora.kernels.loss import GTM, GTN, GTK
from ancora.kernels.fused import _gemm_bf16, _residual_add, RTM, RTN
from ancora.model.qwen3_layer import linear_bf16, _bf

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])

M, H, I, eps = 4096, 1024, 3072, 1e-6
rng = np.random.default_rng(0)
x = (rng.standard_normal((M, H)) * 0.5).astype(np.float32)
w_ln = (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)
w_gate = (rng.standard_normal((H, I)) * 0.02).astype(np.float32)
w_up = (rng.standard_normal((H, I)) * 0.02).astype(np.float32)
w_down = (rng.standard_normal((I, H)) * 0.02).astype(np.float32)


def host_mlp():
    h2, _ = rmsnorm_forward(x, w_ln, si, eps)
    g = linear_bf16(h2, w_gate, si); u = linear_bf16(h2, w_up, si)
    a = swiglu_forward(g, u, si)
    mlp = linear_bf16(a, w_down, si)
    return _bf(x + mlp)


# persistent device buffers
gx = GA(f32bf(x)); gwln = GA(f32bf(w_ln.reshape(1, H)))
gwg = GA(f32bf(w_gate)); gwu = GA(f32bf(w_up)); gwd = GA(f32bf(w_down))
grstd = GA.zeros((M, 1), np.float32); gh2 = GA.zeros((M, H), np.uint16)
gg = GA.zeros((M, I), np.uint16); gu = GA.zeros((M, I), np.uint16)
ga = GA.zeros((M, I), np.uint16); gmlp = GA.zeros((M, H), np.uint16); gout = GA.zeros((M, H), np.uint16)


def resident_mlp(s):
    ct.launch(s, (M // NTM, 1, 1), _rmsnorm_stats, (gx, grstd, H // TH, 1.0 / H, eps))
    ct.launch(s, (M // NTM, 1, 1), _rmsnorm_apply, (gx, gwln, grstd, gh2, H // TH))
    ct.launch(s, (M // GTM, I // GTN, 1), _gemm_bf16, (gh2, gwg, gg, H // GTK, GTM, GTN, GTK))
    ct.launch(s, (M // GTM, I // GTN, 1), _gemm_bf16, (gh2, gwu, gu, H // GTK, GTM, GTN, GTK))
    ct.launch(s, (M // STM, I // TI, 1), _swiglu_fwd, (gg, gu, ga))
    ct.launch(s, (M // GTM, H // GTN, 1), _gemm_bf16, (ga, gwd, gmlp, I // GTK, GTM, GTN, GTK))
    ct.launch(s, (M // RTM, H // RTN, 1), _residual_add, (gx, gmlp, gout))


def wall_ms(fn, it=30, warm=10):
    for _ in range(warm): fn()
    stream.sync(); t = time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter() - t) / it * 1000


if __name__ == "__main__":
    print(f"Device-resident MLP block  M={M} H={H} I={I}")
    print("=" * 60)
    ref = host_mlp()
    resident_mlp(si); cudart.cudaStreamSynchronize(si)
    for nm, buf in [("rmsnorm h2", gh2), ("gate", gg), ("up", gu), ("swiglu act", ga), ("down mlp", gmlp), ("out", gout)]:
        v = bf32(buf.to_numpy())
        print(f"  [stage] {nm:12s} nan={np.isnan(v).any()} max={np.nanmax(np.abs(v)):.3f}")
    devv = bf32(gout.to_numpy())
    rel = np.abs(devv - ref).max() / (np.abs(ref).max() + 1e-9)   # ref is already bf16-valued f32
    print(f"  correctness: device vs host MLP  {rel*100:.3f}%  {'OK' if rel < 0.01 else 'FAIL'}")
    th = wall_ms(host_mlp); td = wall_ms(lambda: (resident_mlp(si), stream.sync()))
    print(f"  host-glue MLP   {th*1000:7.0f} µs")
    print(f"  device-resident {td*1000:7.0f} µs   ({th/td:.0f}x faster)")
    print("=" * 60)
