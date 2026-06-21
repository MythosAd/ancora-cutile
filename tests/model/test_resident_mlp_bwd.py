"""Device-resident MLP BACKWARD (down/gate/up _gemm_dx+_gemm_dW, swiglu_bwd, rmsnorm_bwd,
residual joins) vs host MLP backward. Proves the device-resident backward assembly + speedup."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart

from ancora.kernels.norm import (_rmsnorm_bwd_dx, _rmsnorm_dw_part, _rmsnorm_dw_reduce,
                                  rmsnorm_forward, rmsnorm_backward, TM as NTM, TH, TD, PART,
                                  f32_to_bf16_bits as f32bf, bf16_bits_to_f32 as bf32)
from ancora.kernels.activation import _swiglu_bwd, swiglu_forward, swiglu_backward, TM as STM, TI
from ancora.kernels.fused import _gemm_dx, _gemm_dW, _residual_add, RTM, RTN
from ancora.model.qwen3_layer import linear_bf16, linear_bf16_backward, _bf

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
stream = dev.create_stream(); si = int(stream.__cuda_stream__()[1])
M, H, I, eps, T = 512, 1024, 3072, 1e-6, 64
rng = np.random.default_rng(0)
x2 = _bf((rng.standard_normal((M, H)) * 0.5).astype(np.float32))
w_pln = (1.0 + rng.standard_normal(H) * 0.05).astype(np.float32)
w_gate = (rng.standard_normal((H, I)) * 0.02).astype(np.float32)
w_up = (rng.standard_normal((H, I)) * 0.02).astype(np.float32)
w_down = (rng.standard_normal((I, H)) * 0.02).astype(np.float32)
d_out = _bf((rng.standard_normal((M, H)) * 0.1).astype(np.float32))

# ── host forward (cache) + backward (reference) ──
h2, rstd2 = rmsnorm_forward(x2, w_pln, si, eps)
gate = linear_bf16(h2, w_gate, si); up = linear_bf16(h2, w_up, si); act = swiglu_forward(gate, up, si)
d_a, gW_down = linear_bf16_backward(d_out, act, w_down, si)
d_gate, d_up = swiglu_backward(gate, up, d_a, si)
d_h2a, gW_gate = linear_bf16_backward(d_gate, h2, w_gate, si)
d_h2b, gW_up = linear_bf16_backward(d_up, h2, w_up, si)
d_h2 = d_h2a + d_h2b
d_x2m, gW_pln = rmsnorm_backward(x2, w_pln, d_h2, rstd2, si)
d_x2_ref = _bf(d_out + d_x2m)

# ── device-resident buffers ──
class GA:
    def __init__(s, a):
        s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
def Z(sh, dt=np.uint16): return GA(np.zeros(sh, dt))

gdout = GA(f32bf(d_out)); gact = GA(f32bf(act)); ggate = GA(f32bf(gate)); gup = GA(f32bf(up))
gh2 = GA(f32bf(h2)); gx2 = GA(f32bf(x2)); grstd2 = GA(rstd2.astype(np.float32))
gwdown = GA(f32bf(w_down)); gwgate = GA(f32bf(w_gate)); gwup = GA(f32bf(w_up)); gwpln = GA(f32bf(w_pln.reshape(1, H)))
gda = Z((M, I)); gdg = Z((M, I)); gdu = Z((M, I)); gdh2a = Z((M, H)); gdh2b = Z((M, H)); gdh2 = Z((M, H))
gdx2m = Z((M, H)); gdx2 = Z((M, H))
gGdown = Z((I, H), np.float32); gGgate = Z((H, I), np.float32); gGup = Z((H, I), np.float32)
gGpln = Z((1, H), np.float32); gpart = Z((PART, H), np.float32)
MB = M // NTM; BPP = (MB + PART - 1) // PART


def dx(dy, W, out, K, N): ct.launch(si, (M // T, K // T, 1), _gemm_dx, (dy, W, out, N // T, T, T, T))
def dW(xb, dy, out, K, N): ct.launch(si, (K // T, N // T, 1), _gemm_dW, (xb, dy, out, M // T, T, T, T))


def resident_mlp_bwd():
    dx(gdout, gwdown, gda, I, H);  dW(gact, gdout, gGdown, I, H)      # d_a, dW_down
    ct.launch(si, (M // STM, I // TI, 1), _swiglu_bwd, (ggate, gup, gda, gdg, gdu))
    dx(gdg, gwgate, gdh2a, H, I);  dW(gh2, gdg, gGgate, H, I)         # d_h2a, dW_gate
    dx(gdu, gwup, gdh2b, H, I);    dW(gh2, gdu, gGup, H, I)           # d_h2b, dW_up
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gdh2a, gdh2b, gdh2))
    ct.launch(si, (M // NTM, 1, 1), _rmsnorm_bwd_dx, (gx2, gwpln, gdh2, grstd2, gdx2m, H // TH, 1.0 / H))
    ct.launch(si, (H // TD, PART, 1), _rmsnorm_dw_part, (gx2, gdh2, grstd2, gpart, MB, BPP))
    ct.launch(si, (H // TD, 1, 1), _rmsnorm_dw_reduce, (gpart, gGpln))
    ct.launch(si, (M // RTM, H // RTN, 1), _residual_add, (gdout, gdx2m, gdx2))


if __name__ == "__main__":
    print(f"Device-resident MLP backward  M={M} H={H} I={I}"); print("=" * 60)
    resident_mlp_bwd(); cudart.cudaStreamSynchronize(si)
    rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    checks = [("d_x2", bf32(gdx2.np()), d_x2_ref), ("dW_down", gGdown.np(), gW_down),
              ("dW_gate", gGgate.np(), gW_gate), ("dW_up", gGup.np(), gW_up), ("dW_post_ln", gGpln.np().reshape(H), gW_pln)]
    ok = True
    for nm, a, b in checks:
        e = rel(a, b); o = e < 0.04; ok &= o
        print(f"  {nm:11s} {e*100:.2f}%  {'OK' if o else 'FAIL'}")

    def wall(fn, it=20, warm=5):
        for _ in range(warm): fn()
        stream.sync(); t = time.perf_counter()
        for _ in range(it): fn()
        return (time.perf_counter() - t) / it * 1000
    def host_bwd():
        da, _ = linear_bf16_backward(d_out, act, w_down, si); dg, du = swiglu_backward(gate, up, da, si)
        dha, _ = linear_bf16_backward(dg, h2, w_gate, si); dhb, _ = linear_bf16_backward(du, h2, w_up, si)
        rmsnorm_backward(x2, w_pln, dha + dhb, rstd2, si)
    th = wall(host_bwd); td = wall(lambda: (resident_mlp_bwd(), stream.sync()))
    print(f"  host-glue {th*1000:.0f} µs | device-resident {td*1000:.0f} µs | {th/td:.0f}x faster")
    print("=" * 60); print(f"  {'PASS' if ok else 'FAIL'}")
