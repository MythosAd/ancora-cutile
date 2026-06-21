"""FP8 attention (_attn_fwd_tok_fp8, SageAttention-style) — accuracy vs an f32 reference
(alongside the BF16 kernel) and speed. Decides whether FP8 attention is worth it: attention's
tensor peak is 80 (BF16) vs 184 (FP8) on this GPU, so up to ~2× IF accuracy holds."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, ancora.env
import cuda.core as cc, cuda.tile as ct
from cuda.bindings import driver as cdrv, runtime as cudart
from ancora.kernels.attention import _attn_fwd_tok, _attn_fwd_tok_fp8, BQ, D

cudart.cudaFree(0); dev = cc.Device(0); dev.set_current()
so = dev.create_stream(); SI = int(so.__cuda_stream__()[1])
bf = lambda x: (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
bfv = lambda x: (bf(x).astype(np.uint32) << 16).view(np.float32)
b2f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
rel = lambda a, b: np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)

class GA:
    def __init__(s, a):
        a = np.ascontiguousarray(a); s.sh, s.dt, s.nb = a.shape, a.dtype, a.nbytes
        _, s.p = cdrv.cuMemAlloc(a.nbytes); cdrv.cuMemcpyHtoD(s.p, a, a.nbytes)
        s.__cuda_array_interface__ = {"shape": a.shape, "typestr": a.dtype.str, "data": (int(s.p), False), "version": 3}
    def np(s): o = np.empty(s.sh, s.dt); cdrv.cuMemcpyDtoH(o, s.p, s.nb); return o
    @classmethod
    def z(c, sh, d): return c(np.zeros(sh, d))

def ref_attn(Q, K, V, B, S, Hq, Hkv, scale):
    """f32 causal GQA reference from token-major bf16-rounded Q/K/V (M,H*D)."""
    G = Hq // Hkv; M = B * S
    Qd = bfv(Q).reshape(B, S, Hq, D); Kd = bfv(K).reshape(B, S, Hkv, D); Vd = bfv(V).reshape(B, S, Hkv, D)
    O = np.zeros((B, S, Hq, D), np.float32)
    mask = np.tril(np.ones((S, S), bool))
    for b in range(B):
        for h in range(Hq):
            kvh = h // G
            sc = (Qd[b, :, h, :].astype(np.float64) @ Kd[b, :, kvh, :].T.astype(np.float64)) * scale
            sc = np.where(mask, sc, -1e30); sc -= sc.max(1, keepdims=True)
            p = np.exp(sc); p /= p.sum(1, keepdims=True)
            O[b, :, h, :] = (p @ Vd[b, :, kvh, :].astype(np.float64)).astype(np.float32)
    return O.reshape(M, Hq * D)

def tms(launch, it=40, wm=10):
    for _ in range(wm): launch()
    so.sync(); _, a = cudart.cudaEventCreate(); _, b = cudart.cudaEventCreate(); cudart.cudaEventRecord(a, SI)
    for _ in range(it): launch()
    cudart.cudaEventRecord(b, SI); cudart.cudaEventSynchronize(b)
    return cudart.cudaEventElapsedTime(a, b)[1] / it


def run(B, S, label, timing=False):
    Hq, Hkv = 16, 8; M, qd, kd = B * S, Hq * D, Hkv * D; NQB = S // BQ; scale = 1.0 / math.sqrt(D)
    rng = np.random.default_rng(0)
    Qn = (rng.standard_normal((M, qd)) * 0.5).astype(np.float32)
    Kn = (rng.standard_normal((M, kd)) * 0.5).astype(np.float32)
    Vn = (rng.standard_normal((M, kd)) * 0.5).astype(np.float32)
    Q = GA(bf(Qn)); K = GA(bf(Kn)); V = GA(bf(Vn))
    Obf = GA.z((M, qd), np.uint16); Ofp = GA.z((M, qd), np.uint16)
    Lb = GA.z((M * Hq, 1), np.float32); Lf = GA.z((M * Hq, 1), np.float32)
    abf = lambda: ct.launch(SI, (NQB, B * Hq, 1), _attn_fwd_tok, (Q, K, V, Obf, Lb, NQB, NQB, Hq, Hkv, scale))
    afp = lambda: ct.launch(SI, (NQB, B * Hq, 1), _attn_fwd_tok_fp8, (Q, K, V, Ofp, Lf, NQB, NQB, Hq, Hkv, scale))
    abf(); afp(); cudart.cudaStreamSynchronize(SI)
    if not timing:
        Oref = ref_attn(Qn, Kn, Vn, B, S, Hq, Hkv, scale)
        ob, of = b2f(Obf.np()), b2f(Ofp.np())
        print(f"  {label}: BF16 vs f32-ref  max {rel(ob,Oref)*100:.2f}%  rms {rms(ob,Oref)*100:.2f}%")
        print(f"  {label}: FP8  vs f32-ref  max {rel(of,Oref)*100:.2f}%  rms {rms(of,Oref)*100:.2f}%")
    else:
        fl = 2.0 * B * Hq * S * S * D   # causal ~ half
        tb, tf = tms(abf), tms(afp)
        print(f"  {label}: BF16 {tb*1000:.0f}us {fl/(tb/1e3)/1e12:.0f} TF | FP8 {tf*1000:.0f}us {fl/(tf/1e3)/1e12:.0f} TF | {tb/tf:.2f}x")


if __name__ == "__main__":
    print("FP8 attention — accuracy + speed"); print("=" * 64)
    print("accuracy (vs f32 reference):")
    run(1, 256, "S=256 ")
    run(2, 512, "S=512 ")
    print("speed (M=8192):")
    run(4, 2048, "S=2048", timing=True)
    print("=" * 64)
